from src import housing_module

housing_module.fetch_housing_data()
data=housing_module.load_housing_data()
print(data.head())



prep,lab,testset=housing_module.preprocess(data)

lin_reg,tree_reg,forest_reg,grid_search=housing_module.train(prep,lab)

lin_scores,tree_scores,forest_scores,grid_search_scores = housing_module.scoring(data,lin_reg,tree_reg,forest_reg,grid_search)
print(lin_scores,tree_scores,forest_scores,grid_search_scores)