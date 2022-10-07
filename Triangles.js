/// <reference path="TSDef/p5.global-mode.d.ts" />

let triangles;
let solver;
let history;
let iteration;
const MAX_ITERATION = 1500; //1000;
const NSHAPES = 50;
const NPOPULATION = 100; //100
const IMG_SIZE = 200;
const RAND_SEED = 999;
let targetImg;
let canvasPixels;
let diffImg;

function preload() {
  targetImg = loadImage('assets/silly_cat.png');
}

function setup() {
  createCanvas(IMG_SIZE * 3, IMG_SIZE);
  //colorMode(RGB);
  noStroke();
  randomSeed(RAND_SEED);
  triangles = new TrianglesPainter(IMG_SIZE, IMG_SIZE, NSHAPES, 0.5, 1.0, true);
  solver = new PGPE(triangles.n_params);
  history = [];
  iteration = 0;
  targetImg.loadPixels();
  console.log('Init MB used: ' + tf.memory().numBytesInGPU/1024/1024 + ' for numTensors: ' + tf.memory().numTensors);
}

function draw() {
  let fitness_list, result, solutions;
  if (iteration < MAX_ITERATION) {
      // solutions = tf.tidy(() => {return solver.ask();});
      // fitness_list = [];
      // for (let i = 0; i < solver.popsize; i += 1) {
      //     fitness_list[i] = tf.tidy(() => {return fit_func(solutions.unstack()[i]);});
      // }
      // result = tf.tidy(() => {return solver.tell(fitness_list);});
      console.log('Tell MB used: ' + tf.memory().numBytesInGPU/1024/1024 + ' for numTensors: ' + tf.memory().numTensors);
      // history.push(result[1]);
      // if ((((iteration + 1) % 1) === 0)) { //100
      //     console.log("fitness at iteration", (iteration + 1), ": ", result[1]);
      // }
      // //print best
      // tf.tidy(() => {return fit_func(result[0]);});
      // solutions.dispose();
      tf.tidy(() => {return solver.iterate();});
      iteration += 1;
      if (iteration == MAX_ITERATION) {
        saveCanvas('final image', 'png');
      }
  }
  //console.log("local optimum discovered by solver:\n", result[0]);
  //console.log("fitness score at this local optimum:", result[1]);
}

function fit_func(params) {
  triangles.render(params, "white");
  canvasPixels = get(0,0,IMG_SIZE,IMG_SIZE);
  diffImg = canvasPixels;
  diffImg.blend(targetImg,0,0,IMG_SIZE,IMG_SIZE,0,0,IMG_SIZE,IMG_SIZE,DIFFERENCE);
  image(diffImg, IMG_SIZE * 2, 0);
  diffImg.loadPixels();
  let l2loss = diffImg.pixels.filter((_,idx) => idx  % 4 != 3).reduce((a,b) => a + Math.pow(b/255,2), 0) / (IMG_SIZE * IMG_SIZE * 3);
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
    solutions = solver.ask(); //tf.tidy(() => {return solver.ask();});
    fitness_list = [];
    for (let i = 0; i < solver.popsize; i += 1) {
        fitness_list[i] = fit_func(solutions.unstack()[i]); //tf.tidy(() => {return fit_func(solutions.unstack()[i]);});
    }
    result = solver.tell(fitness_list); //tf.tidy(() => {return solver.tell(fitness_list);});
    //console.log('Tell MB used: ' + tf.memory().numBytesInGPU/1024/1024 + ' for numTensors: ' + tf.memory().numTensors);
    history.push(result[1]);
    if ((((iteration + 1) % 1) === 0)) { //100
        console.log("fitness at iteration", (iteration + 1), ": ", result[1]);
    }
    //print best
    fit_func(result[0]); //tf.tidy(() => {return fit_func(result[0]);});
    //solutions.dispose();

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
    //let sorted = x.slice().sort(function(a,b){return b-a})
    let ranks = Array.from({length: x.length}, (_, i) => i);
    //let ranks = x.map(function(v){ return sorted.indexOf(v)+1 });
    ranks.sort((a, b) => x.indexOf(a) - x.indexOf(b));
    //var ranks;
    //_pj._assert((x.ndim === 1), null);
    //ranks = np.empty(x.length, {"dtype": "int"});
    //ranks[x.argsort()] = np.arange(x.length);
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
    // weight_decay = this.weight_decay = scalar (0.01)
    // model_param_list = this.solutions = tensor1d
    //let model_param_grid;
    //console.log(model_param_list);
    //model_param_grid = tf.tensor2d(model_param_list);
    let decay = model_param_list.mul(model_param_list).mean(1).mul(-weight_decay).arraySync();
    return x.map((a, i) => a + decay[i]);
    //return ((- weight_decay) * np.mean((model_param_grid * model_param_grid), {"axis": 1}));
  }
  
  result() {
      return [this.best_mu, this.best_reward, this.curr_best_reward, this.sigma];
  }
}

class TrianglesPainter{
  constructor(h, w, n_triangle = 10, alpha_scale = 0.1, coordinate_scale = 1.0, normalise=true) {
    this.h = h;
    this.w = w;
    this.n_triangle = n_triangle;
    this.alpha_scale = alpha_scale;
    this.coordinate_scale = coordinate_scale;
    this.normalise = normalise;
  }

  get n_params() {
    return this.n_triangle * 10;
  }

  normaliseSlice(slice) {
    if (this.normalise) {
      return slice.sub(slice.min()).div(slice.max().sub(slice.min())).flatten().arraySync();
    }
    else {
      return slice.add(0.5).flatten().arraySync();
    }
  }


  render(params, canvas_background = "noise") {
    let a, alpha_scale, b, coordinate_scale, g, h, n_triangle, r, w, x0, x1, x2, xc, y0, y1, y2, yc;
    [h, w] = [this.h, this.w];
    alpha_scale = this.alpha_scale;
    coordinate_scale = this.coordinate_scale;
    params = params.reshape([-1, 10]);
    n_triangle = params.shape[0];

    // normalise data
    const arr_x0 = this.normaliseSlice(params.slice([0, 0], [-1, 1]));
    const arr_y0 = this.normaliseSlice(params.slice([0, 1], [-1, 1]));
    const arr_x1 = this.normaliseSlice(params.slice([0, 2], [-1, 1]));
    const arr_y1 = this.normaliseSlice(params.slice([0, 3], [-1, 1]));
    const arr_x2 = this.normaliseSlice(params.slice([0, 4], [-1, 1]));
    const arr_y2 = this.normaliseSlice(params.slice([0, 5], [-1, 1]));
    const arr_r = this.normaliseSlice(params.slice([0, 6], [-1, 1]));
    const arr_g = this.normaliseSlice(params.slice([0, 7], [-1, 1]));
    const arr_b = this.normaliseSlice(params.slice([0, 8], [-1, 1]));
    const arr_a = this.normaliseSlice(params.slice([0, 9], [-1, 1]));

    if (canvas_background === "noise") {
      // TODO - random RGB(255,255,255) per pixel, preferably used with repeated evaluations to get average less affected by noise itself
    } else {
      if (canvas_background === "white") {
        background(255);;
      } else {
        if (canvas_background === "black") {
          background(0);
        }
      }
    }

    for (let i = 0; i < n_triangle; i += 1) {
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
      [xc, yc] = [(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0];
      [x0, y0] = [xc + (x0 - xc) * coordinate_scale, yc + (y0 - yc) * coordinate_scale];
      [x1, y1] = [xc + (x1 - xc) * coordinate_scale, yc + (y1 - yc) * coordinate_scale];
      [x2, y2] = [xc + (x2 - xc) * coordinate_scale, yc + (y2 - yc) * coordinate_scale];
      [x0, x1, x2] = [Number.parseInt(x0 * w), Number.parseInt(x1 * w), Number.parseInt(x2 * w)];
      [y0, y1, y2] = [Number.parseInt(y0 * h), Number.parseInt(y1 * h), Number.parseInt(y2 * h)];
      [r, g, b, a] = [Number.parseInt(r * 255), Number.parseInt(g * 255), Number.parseInt(b * 255), Number.parseInt(a * alpha_scale * 255)];

      fill(r, g, b, a);
      triangle(x0, y0, x1, y1, x2, y2);
    }
  }

}
