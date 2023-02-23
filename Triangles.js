/// <reference path="TSDef/p5.global-mode.d.ts" />

let triangles;
let solver;
let history;
let iteration;
let startTime;
const MAX_ITERATION = 1500; //1000;
const STOP_SCALING_AFTER = 200; //stop scaling all triangles to (varying) most extreme triangles to give stability
const NSHAPES = 50;
const NPOPULATION = 99 + 1; //100. The +1 is for the background colour, total needs to be even
const IMG_SIZE = 200;
const RAND_SEED = 999;
let targetImg;
let targetPixels;
let canvasPixels;
let diffImg;
let offscreenTriangles;
let loadedFont;

function preload() {
  targetImg = loadImage('assets/silly_cat.png');
  loadedFont = loadFont("assets/Arial.ttf");
}

function setup() {
  pixelDensity(1);
  createCanvas(IMG_SIZE * 3, IMG_SIZE, WEBGL);
  ortho(0, IMG_SIZE * 3, 0, -IMG_SIZE, 0, NPOPULATION * 2); //Not sure why need z axis related to number of triangles by this factor, just from experimenting
  offscreenTriangles = createGraphics(IMG_SIZE, IMG_SIZE, WEBGL);
  offscreenTriangles.noStroke();
  offscreenTriangles.ortho(0, IMG_SIZE, 0, -IMG_SIZE, 0, 200);
  textFont(loadedFont);
  noStroke();
  //randomSeed(RAND_SEED); //For debugging
  triangles = new TrianglesPainter(IMG_SIZE, IMG_SIZE, NSHAPES, 0.5, 1.0, true);
  solver = new PGPE(triangles.n_params);
  history = [];
  iteration = 0;
  targetImg.loadPixels();
  targetPixels = tf.browser.fromPixels(targetImg.canvas).flatten();
  console.log(`Init used ${(tf.memory().numBytesInGPU/1024/1024).toFixed(3)}MB for ${tf.memory().numTensors} Tensors`);
}

function draw() {
  if (iteration < MAX_ITERATION) {
      console.log(`Tell used ${(tf.memory().numBytesInGPU/1024/1024).toFixed(3)}MB for ${tf.memory().numTensors} Tensors`);
      background(255);
      startTime = millis();
      tf.tidy(() => {return solver.iterate();});
      iteration += 1;
      fill(0);
      text("Iteration " + iteration + " of " + MAX_ITERATION, IMG_SIZE * 2 + 10, 10);
      if (iteration == MAX_ITERATION) {
        saveCanvas('final image', 'png');
      }
  }
}

function fit_func(params, show=false) {
  //debugger;
  triangles.render(params, "evolved");
  diffImg = tf.browser.fromPixels(offscreenTriangles.canvas).flatten();

  let l2loss = 0;
  l2loss = tf.squaredDifference(targetPixels, diffImg).sum().arraySync();
  l2loss /= 255 * 255 * IMG_SIZE * IMG_SIZE * 3;
  if (show) {
    image(offscreenTriangles, 0, 0);
    image(targetImg, IMG_SIZE, 0);
  }
  return 1 - l2loss; //pgpe maximises
}


class Adam {
  constructor(pi, stepsize, beta1 = 0.99, beta2 = 0.999, epsilon = 1e-08) {
    this.pi = pi;
    this.dim = pi.num_params;
    this.epsilon = epsilon;
    this.t = 0;
    this.stepsize = stepsize;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.m = tf.zeros([this.dim], 'float32');
    this.v = tf.zeros([this.dim], 'float32');
  }

  update(globalg) {
    let ratio, step;
    this.t += 1;
    step = this._compute_step(globalg);
    ratio = tf.norm(step).div((tf.norm(this.pi.mu).add(this.epsilon)));
    this.pi.mu = tf.keep(this.pi.mu.add(step));
    return ratio;
  }

  _compute_step(globalg) {
    let a, step;
    a = -this.stepsize * Math.sqrt(1 - Math.pow(this.beta2, this.t)) / (1 - Math.pow(this.beta1, this.t));
    this.m = tf.keep(tf.mul(this.beta1, this.m).add(tf.mul(1 - this.beta1, globalg)));
    this.v = tf.keep(tf.mul(this.beta2, this.v).add(tf.mul(1 - this.beta2, tf.mul(globalg, globalg))));
    step = tf.mul(a, this.m.div((tf.sqrt(this.v).add(this.epsilon))));
    return step;
  }

}

class ClipUp {
  constructor(pi, solution_length, stepsize, momentum = 0.9, max_speed = 0.15, fix_gradient_size = true) {
    this.pi = pi;
    this.dim = pi.num_params;
    this.solution_length = solution_length;
    this.momentum = momentum;
    this.t = 0;
    this.stepsize = stepsize;
    this.max_speed = max_speed;
    this.fix_gradient_size = fix_gradient_size;
    //this.m = tf.zeros([this.dim], 'float32');
    this.v = tf.zeros([this.dim], 'float32');
  }

  update(globalg) {
    let ratio, step;
    this.t += 1;
    step = this._compute_step(globalg);
    //ratio = tf.norm(step).div((tf.norm(this.pi.mu).add(this.epsilon)));
    ratio = 1;
    this.pi.mu = tf.keep(this.pi.mu.add(step));
    return ratio; //not used
  }

  clip(x, max_length) {
    var length, ratio;
    length = tf.norm(x);

    if (length > max_length) {
      ratio = max_length / length;
      return x.mul(ratio);
    } else {
      return x;
    }
  }

  _compute_step(globalg) {
    var g_len, step;

    if (this.fix_gradient_size) {
      g_len = tf.norm(globalg);
      globalg = globalg.div(g_len);
    }

    step = globalg.mul(this.stepsize);
    this.v = this.v.mul(this.momentum).add(step);
    this.v = tf.keep(this.clip(this.v, this.max_speed));
    return this.v.neg();
  }
}


class PGPE{
  constructor(num_params,       // number of model parameters
  sigma_init=0.50,              // initial standard deviation
  sigma_alpha=0.20,             // learning rate for standard deviation
  sigma_decay=0.999,            // anneal standard deviation
  sigma_limit=0.01,             // stop annealing if less than this
  sigma_max_change=0.2,         // clips adaptive sigma to 20%
  learning_rate=0.1,            // learning rate for standard deviation
  learning_rate_decay = 1.0,    // annealing the learning rate
  learning_rate_limit = 0.01,   // stop annealing learning rate
  elite_ratio = 0,              // if > 0, then ignore learning_rate
  popsize=NPOPULATION,          // population size
  average_baseline=true,        // set baseline to average of batch
  weight_decay=0.00,            // weight decay coefficient
  rank_fitness=false,           // use rank rather than fitness numbers //TODO - buggy if set to true, details on 'best fitness' over time are meaningless
  forget_best=false) {           // don't keep the historical best solution) 
    
    this.num_params = num_params;
    this.sigma_init = sigma_init;
    this.sigma_alpha = sigma_alpha;
    this.sigma_decay = sigma_decay;
    this.sigma_limit = sigma_limit;
    this.sigma_max_change = sigma_max_change;
    this.learning_rate = learning_rate;
    this.learning_rate_decay = learning_rate_decay;
    this.learning_rate_limit = learning_rate_limit;
    this.popsize = popsize;
    this.average_baseline = average_baseline;
    if (this.average_baseline) {
      this.batch_size = Number.parseInt(this.popsize / 2);
    } else {
      this.batch_size = Number.parseInt((this.popsize - 1) / 2);
    }
    this.elite_ratio = elite_ratio;
    this.elite_popsize = Number.parseInt(this.popsize * this.elite_ratio);
    this.use_elite = false;

    if (this.elite_popsize > 0) {
      this.use_elite = true;
    }

    this.forget_best = forget_best;
    this.mu = tf.zeros([this.num_params]);
    this.sigma = tf.ones([this.num_params]).mul(this.sigma_init);
    this.curr_best_mu = tf.zeros([this.num_params]);
    this.best_mu = tf.zeros([this.num_params]);
    this.best_reward = 0;
    this.first_iteration = true;
    this.weight_decay = weight_decay;
    this.rank_fitness = rank_fitness;

    if (this.rank_fitness) {
      this.forget_best = true;
    }

    //this.optimizer = new Adam(this, learning_rate);
    this.optimizer = new ClipUp(this, triangles.n_params, learning_rate);
    this.seed = RAND_SEED;
  }

  iterate() {
    let fitness_list, result, solutions;
    solutions = solver.ask();
    //debugger;
    fitness_list = [];
    for (let i = 0; i < solver.popsize; i += 1) {
        fitness_list[i] = fit_func(solutions.slice([i, 0], [1, -1]));
    }
    result = solver.tell(fitness_list);
    //console.log('Tell MB used: ' + tf.memory().numBytesInGPU/1024/1024 + ' for numTensors: ' + tf.memory().numTensors);
    history.push(result[1]);
    let endTime = millis();
    let timeForIteration = endTime - startTime;
    startTime = endTime;
    if ((((iteration + 1) % 1) === 0)) { //100
        console.log(`fitness at iteration ${iteration + 1}: ${result[1].toFixed(8)} in ${timeForIteration.toFixed(0)}ms`);
    }
    //print best
    fit_func(result[0], true);
  }


  ask() {
    //returns a list of parameters
    //antithetic sampling
    if (!this.first_iteration) {
      this.epsilon.dispose();
      this.epsilon_full.dispose();
      this.solutions.dispose();
    }
    this.epsilon = tf.randomNormal([this.batch_size, this.num_params], 0, 1, 'float32', this.seed).mul(this.sigma.reshape([1, this.num_params]));
    this.seed += 1;
    this.epsilon_full = this.epsilon.concat(tf.neg(this.epsilon));
    if (this.average_baseline) {
      this.solutions = this.mu.reshape([1, this.num_params]).add(this.epsilon_full);
    }
    else {
      //first population is mu, then positive epsilon, then negative epsilon
      this.solutions = this.mu.reshape([1, this.num_params]).add(tf.zeros([1, this.num_params]).concat(this.epsilon_full));
    }
    return this.solutions;
  }

  tell(reward_table_result) {
    let S, b, best_mu, best_reward, change_mu, change_sigma, delta_sigma, idx, r, rS, rT, reward, reward_avg, reward_offset, reward_table, stdev_reward, update_ratio;
    reward_table = reward_table_result;
    if (this.rank_fitness) {
        reward_table = this.compute_centered_ranks(reward_table);
    }
    if ((this.weight_decay > 0)) {
        reward_table = this.apply_weight_decay(reward_table, this.weight_decay, this.solutions);
    }
    reward_offset = 1;
    if (this.average_baseline) {
        b = reward_table.reduce((current, runningTotal) => current + runningTotal) / reward_table.length;
        reward_offset = 0;
    } else {
        b = reward_table[0];
    }

    //helpers to create equivalent of np.argsort()
    let decor = (v, i) => [v, i];          // set index to value
    let undecor = a => a[1];               // leave only index
    let argsort = arr => arr.map(decor).sort().map(undecor);

    reward = reward_table.slice(reward_offset);
    if (this.use_elite) {
        idx = argsort(reward).reverse().slice(0, this.elite_popsize);
    } else {
        idx = argsort(reward).reverse();
    }
    best_reward = reward[idx[0]];
    if (((best_reward > b) || this.average_baseline)) {
        best_mu = this.mu.add(this.epsilon_full.unstack()[idx[0]]);
        best_reward = reward[idx[0]];
    } else {
        best_mu = this.mu;
        best_reward = b;
    }
    this.curr_best_reward = best_reward;
    this.curr_best_mu = best_mu;
    if (this.first_iteration) {
        this.first_iteration = false;
        this.best_reward = this.curr_best_reward;
        this.best_mu = tf.keep(best_mu);
    } else {
        if ((this.forget_best || (this.curr_best_reward > this.best_reward))) {
            this.best_mu = tf.keep(best_mu);
            this.best_reward = this.curr_best_reward;
        }
    }
    r = tf.tensor1d(reward);
    if (this.use_elite) {
        this.mu = this.mu.add(this.epsilon_full[idx].mean({"axis": 0}));
    } else {
        rT = r.slice(0, this.batch_size).sub(r.slice(this.batch_size));
        change_mu = rT.dot(this.epsilon);
        this.optimizer.stepsize = this.learning_rate;
        update_ratio = this.optimizer.update(change_mu.neg());
    }
    if ((this.sigma_alpha > 0)) {
        stdev_reward = 1.0;
        if ((! this.rank_fitness)) {
            stdev_reward = tf.moments(r).variance.sqrt();
        }
        S = (tf.mul(this.epsilon, this.epsilon).sub(tf.mul(this.sigma, this.sigma).reshape([1, this.num_params]))).div(this.sigma.reshape([1, this.num_params]));
        reward_avg = r.slice(0, this.batch_size).add(r.slice(this.batch_size)).div(2.0);
        rS = reward_avg.sub(b);
        delta_sigma = tf.dot(rS, S).div(stdev_reward.mul(2 * this.batch_size)).mul(this.sigma_alpha);
        change_sigma = tf.minimum(tf.maximum(delta_sigma, this.sigma.mul(this.sigma_max_change).neg()), this.sigma.mul(this.sigma_max_change)); //BUG - need to use tf.maximum/minium as clipbyvalue uses single scalar values, not arrays
        this.sigma = this.sigma.add(change_sigma);
    }
    if ((this.sigma_decay < 1)) {
      this.sigma = tf.keep(this.sigma.mul(this.sigma_decay).where(this.sigma.greater(this.sigma_limit), this.sigma));
    }
    if (((this.learning_rate_decay < 1) && (this.learning_rate > this.learning_rate_limit))) {
        this.learning_rate *= this.learning_rate_decay;
    }
    S.dispose();
    change_sigma.dispose();
    change_mu.dispose();
    delta_sigma.dispose();
    r.dispose();
    rT.dispose();
    rS.dispose();
    reward_avg.dispose();
    return [this.best_mu, this.best_reward, this.curr_best_reward, this.sigma];
  }

  compute_ranks(x) {
    /*
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    */
    let ranks = Array.from({length: x.length}, (_, i) => i);
    ranks.sort((a, b) => x.indexOf(a) - x.indexOf(b));
    return ranks;
  }

  compute_centered_ranks(x) {
    /*
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    */
    let y;
    y = this.compute_ranks(x); //(x.ravel()).reshape(x.shape).astype(np.float32);
    y = y.map((a) => a / (x.length - 1) - 0.5);
    return y;
  }

  apply_weight_decay(x, weight_decay, model_param_list) {
    let decay = model_param_list.mul(model_param_list).mean(1).mul(-weight_decay).arraySync();
    return x.map((a, i) => a + decay[i]);
  }
  
  result() {
      return [this.best_mu, this.best_reward, this.curr_best_reward, this.sigma];
  }
}

class TrianglesPainter{
  constructor(h, w, n_triangle = 10, alpha_scale = 0.1, normalise=true) {
    this.h = h;
    this.w = w;
    this.n_triangle = n_triangle;
    this.alpha_scale = alpha_scale;
    this.normalise = normalise;
    this.minimums;
    this.maximums;
    this.frozenMinimums;
    this.frozenMaximums;
  }

  get n_params() {
    return this.n_triangle * 10;
  }

  normaliseSlice(slice, specificScalar, use_frozen_scale=false, slice_index=0) {
    if (this.normalise) {
      if (use_frozen_scale)
      {
        return slice.sub(this.frozenMinimums[slice_index]).div((this.frozenMaximums[slice_index] - this.frozenMinimums[slice_index]) / specificScalar).floor().flatten().arraySync();
      }
      else
      {
        return slice.sub(this.minimums[slice_index]).div((this.maximums[slice_index] - this.minimums[slice_index]) / specificScalar).floor().flatten().arraySync();
      }
    }
    else {
      return slice.add(0.5).flatten().arraySync();  //NOT IMPLEMENTED ALL FEATURES - MAYBE DELETE?
    }
  }


  render(params, canvas_background = "noise") {
    let a, alpha_scale, b, g, h, n_triangle, r, w, x0, x1, x2, y0, y1, y2;
    [h, w] = [this.h, this.w];
    alpha_scale = this.alpha_scale;
    params = params.reshape([-1, 10]);
    n_triangle = params.shape[0];

    // normalise data
    this.minimums = params.min(0).arraySync();
    this.maximums = params.max(0).arraySync();
    if (iteration == STOP_SCALING_AFTER) //TODO - not quite right, takes scale from last test case of this iteration rather than selected center?
    {
      this.frozenMinimums = this.minimums;
      this.frozenMaximums = this.maximums;
    }
    const arr_x0 = this.normaliseSlice(params.slice([0, 0], [-1, 1]), w, iteration>STOP_SCALING_AFTER, 0);
    const arr_y0 = this.normaliseSlice(params.slice([0, 1], [-1, 1]), h, iteration>STOP_SCALING_AFTER, 1);
    const arr_x1 = this.normaliseSlice(params.slice([0, 2], [-1, 1]), w, iteration>STOP_SCALING_AFTER, 2);
    const arr_y1 = this.normaliseSlice(params.slice([0, 3], [-1, 1]), h, iteration>STOP_SCALING_AFTER, 3);
    const arr_x2 = this.normaliseSlice(params.slice([0, 4], [-1, 1]), w, iteration>STOP_SCALING_AFTER, 4);
    const arr_y2 = this.normaliseSlice(params.slice([0, 5], [-1, 1]), h, iteration>STOP_SCALING_AFTER, 5);
    const arr_r = this.normaliseSlice(params.slice([0, 6], [-1, 1]), 255.99);
    const arr_g = this.normaliseSlice(params.slice([0, 7], [-1, 1]), 255.99);
    const arr_b = this.normaliseSlice(params.slice([0, 8], [-1, 1]), 255.99);
    const arr_a = this.normaliseSlice(params.slice([0, 9], [-1, 1]), 255.99 * alpha_scale);

      // TODO - set "noise" to random RGB(255,255,255) per pixel, preferably used with repeated evaluations to get average less affected by noise itself
    if (canvas_background === "evolved") {
      offscreenTriangles.background([arr_r[0], arr_g[0], arr_b[0]])
    } else {
      if (canvas_background === "white") {
        offscreenTriangles.background(255);;
      } else {
        if (canvas_background === "black") {
          offscreenTriangles.background(0);
        }
      }
    }

    for (let i = 1; i < n_triangle; i += 1) {
      x0 = arr_x0[i];
      y0 = arr_y0[i];
      x1 = arr_x1[i];
      y1 = arr_y1[i];
      x2 = arr_x2[i];
      y2 = arr_y2[i];
      r = arr_r[i];
      g = arr_g[i];
      b = arr_b[i];
      a = arr_a[i];

      offscreenTriangles.fill(r, g, b, a);
      offscreenTriangles.triangle(x0, y0, x1, y1, x2, y2);
    }
  }

}
