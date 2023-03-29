num_leaves = 10
min_data_in_leaf = 20
sub_feature = 0.5
max_depth = 10

n_estimators = 20000
early_stopping_rounds = 200
bagging_fraction = 0.7
bagging_seed = 0
num_threads = 4
colsample_bytree = 0.7

objective = 'regression'
boosting = 'gbdt'
num_boost_round = 100
learning_rate = 0.001
# device = 'gpu'

metric = 'mae'