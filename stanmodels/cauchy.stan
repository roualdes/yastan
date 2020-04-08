data {
  real<lower=0> nu;
  real<lower=0> s;
}
parameters {
  real x_a;
  real<lower=0> x_b;
}

transformed parameters {
  real x = x_a * sqrt(x_b);
}

model {
  x_a ~ normal(0, 1);
  x_b ~ inv_gamma(0.5, 0.5);
}
