data {
  vector[2] mu;
  matrix[2, 2] S;
}
parameters {
  vector[2] x;
}
model {
  x ~ multi_normal(mu, S);
}
