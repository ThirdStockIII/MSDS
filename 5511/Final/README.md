# House Prices - Advanced Regression Techniques

When looking for inspiration for this Deep Learning project, I found a list of potential project ideas that had datasets and challenges associated with the parameters of this class. There was potential for me to learn about Stock Price Prediction, Weather forecasting, and even a Pneumonia detector. What I ended up feeling inspired by most was to take on the House Prices Model using Advanced Regression Techniques. My girlfriend and I are currently looking at houses, so this topic felt very interesting for what else is going in on my life outside of this class. Additionally, when I was looking at the Kaggle Project, the data comes from home prices in Ames, IA. I did my undergrad at Iowa State University and that is in Ames, IA so I was extremely excited that it felt like this project idea was perfect for me.

## Problem

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

The problem is that it might seem that prices of homes can be unpredicatable, so the goal is to predict the sales price for each house.

## Data

The data comes from a Kaggle competition. It features 4 files; trainc.csv, test.csv, data_description.txt, and sample submission.csv. There are 80 unique columns in this dataset, all of them represent a different aspect of a house, such as the year it was built, the basement size, if the driveway is paved, the roof material, and even information on the neighborhood the house is in. There are a ton of different aspects of a home, so it is of no suprise that there are so many values that can be entered in this dataset.

Anna Montoya and DataCanary. House Prices - Advanced Regression Techniques. https://kaggle.com/competitions/house-prices-advanced-regression-techniques, 2016. Kaggle.

The data has 1459 rows, 80 columns, is 957.39 kB, and is a mixture of integer and object data.

## Exploratory Data Analysis (EDA) - Inspect, Visualize, and Clean the Data

There were a lot of different variables to explore for this data. 80 unique columns makes it challenging to try and focus on what is most important for this analysis. Expanding on that, homes have a tendancy to be wildly different from each other so there are expecting to be some outliers in the data. 

### Outliers

The goal of this section is to plot some of the home features to the Sale Price and to then eliminate any obvious outliers from the data. Anytime there is an obvious outlier, I would query the data to get the ID of that point, and then keep track of it so that I can drop it from the datase as outliers would not be good for the prediction.

### Null Values

I also was motivated to change null values. There weren't any missing variables, but some columns are filled with null. Pool Quality had a lot of null values so it made sense to just remove that column as it isn't frequent enough to include. Misc Features was also a bit weird since they were tracking items such as elevators or tennis courts. This category felt like an outlier in itself so it also made sense to drop this column. Other columns that were similar to these were also dropped.

## Models

Project had goals of making predictions through advanced regression. The model that I will eventually conclude with is going to be a stacked regression model. I will be creating several regression models such as linear, random forest, xgb, ridge, gradient boosting, lgbm, catboosting, and voting. I will then have the best of those models be combined in a stacked regression model and the score I recieve from there will be my results.

Linear Regression -	9.510501616810296e+16
Random Forest Regression - 0.1339500658461279
XGB Regression - 0.11965682811517063
Ridge - 0.10894022173825084
Gradient Boosting Regression - 0.11333215220060736
LGBM Regression - 0.1275678077793979
Cat Boose Regression - 0.11313058072563338
Voting Regression - 0.11956123663283695

Those are the scores that I received from each of the models.

## Results and Analysis:

The goal of the project was to compare the models with Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. With that in mind, the best-performing regression models (Gradient Boosting, XGBoost, CatBoost, LightGBM, Random Forest).

With those models, the next step of making a solid prediction model was to implement Stacked Regression. Stacking approach often leads to improved predictive performance compared to any of the individual base models and the final result was a RMSE of 0.11966001781311514.

## Conclusion

I had a lot of fun with this project. It really felt like it was calling me between the life situation I am currently in, and the fact that the database was from a place I lived for a chunk of my life. I learned a lot doing the project too. I really was dragging my feet when I was starting the assignment, because I don't have a lot of confidence with Machine Learning topics, but it was really fun once I got started with the work. I learned new regression techniques which was also rewarding.

One honest takeaway was that I wasn't sure how deep learning this was. Like it felt like a supervised learning project, but it was listed as a potential project and the assignment guidelines mention doing a regression project counts. But all of the other projects had neural networks and this was just regression. I really hope this is a deep learning project because the stacked regression model did take some time to run which makes me feel like it falls under the umbrella of deep learning, but I am not sure entirely.

Hope everyone the best of luck in their classes, I believe in you all.
