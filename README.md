 ## 1	Overview
### 1.1	Description of Project

This is a prediction problem  project for Walmart (A Top Retail Group) Sales dataset from Kaggle for the unit sales forecasting. Advanced and comprehensive analytics skills, including Exploratory Data Analysis and Machine Learning Data Prediction Analysis techniques will be used in this case for generating data-driven business insights.

### 1.2	Business Context

In the dynamic landscape of the retail industry, the ability to predict sales accurately is paramount for sustaining and enhancing business operations. For a retail giant like Walmart, whose vast operations span a multitude of products, locations, and customer segments, the challenge of forecasting sales becomes even more intricate.
Challenges and Risks:
Walmart confronts the formidable task of maximizing decision-making efficiency amid a sea of data. The stakes are high, as inaccurate predictions can lead to substantial losses. Traditional prediction methods, once reliable, now struggle to cope with the complexities of modern retail dynamics. To avoid costly mistakes and enhance forecasting accuracy, there is a pressing need for the integration of cutting-edge data science techniques.
Business Imperatives:
Precise sales predictions stand as the linchpin in Walmart's strategy to navigate both realized and potential revenue opportunities. Efficient inventory management, customer satisfaction, strategic promotions, and a competitive edge hinge on the ability to foresee market trends accurately.
Benefits of Sales Prediction:\
(1)	Efficient Inventory Management: Anticipate demand trends, reducing stockouts and overstocks.\
(2)	Customer Satisfaction: Ensure product availability, meeting customer expectations.\
(3)	Smart Promotions: Strategically plan promotions based on predictive insights.\
(4)	Competitive Edge: Stay ahead by responding swiftly to market shifts.\
(5)	Optimized Supply Chain: Streamline operations for cost-effective supply chain management.\
(6)	Support for Strategic Decisions: Informed decision-making for sustained growth.\

(7)	Reduce Financial Risks: Improve budget management efficiency through accurate sales forecasts.\
(8)	Raise Shareholder Confidence: Provide stakeholders with reliable projections, enhancing trust.

**Situation:**
Walmart is at the nexus of leveraging its rich dataset to drive decision-making efficiency. The precision of sales predictions becomes pivotal, steering the company away from both tangible and missed revenue opportunities.
Key Question:
How can Walmart forecast daily sales for the next 28 days, leveraging hierarchical sales data effectively?
Proposed Solution:
The proposed solution involves harnessing the power of machine learning to predict future sales. By embracing advanced analytics, we  aim to enhance the  forecast accuracy for Walmart, ensuring a proactive and data-driven approach to sales management.

This strategic integration of data science not only addresses current challenges but positions Walmart at the forefront of innovative and efficient retail practices, fostering sustained growth and market leadership.

## 2. Data Description & Exploratory Data Analysis

<img width="670" alt="Screenshot 2024-01-04 at 19 59 54" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/ca4cde95-c4f4-4399-8041-071ab7ac8683">

This table shows the overview of the Input Data:
Raw Data	Description	# Feature	# Record	Data Size
calendar.csv	Workday & Special event day (e.g. SuperBowl)	14	1.9 K	103 kB 
sell_prices.csv	Price of the products sold per store and date	4	6.84 M	203.4 MB
sales_train_validation.csv	historical daily unit sales data per product and store [d1 - d1913]	1019	30.5 K	120 MB
sales_train_evaluation.csv	sales [d1 - d_1941]	1047	30.5 K	121.7 MB
Based on the structure of data we see the data would be of below format:

<img width="685" alt="Screenshot 2024-01-04 at 20 00 36" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/d5acb20b-a021-4245-8f3d-27e581c4b1a9">


## 3. Methodology 


<img width="660" alt="Screenshot 2024-01-04 at 22 49 07" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/56702eff-a1f8-4fbe-ad85-ceb02ba3dde6">





**3.3.1. Price feature**\
We are doing feature engineering here to get price related data, we have week wise data of price (we have price features for test weeks as well).
We are using expanding max price , minimum price , standard deviation , mean, so that there is no data   leakage from future to past, and ,model can solely use the past data using expanding method (since the data is already sorted time wise we are not sorting again , saves in computation time).


**3.3.2. Calendar features**\
we see prices of some items starting for a particular week, which might indicate that would be release week for the product so we can use data in base data frame after that point (as since earlier data was in long format it would have data for all items through all days)
This reduces the size of the data and will have features of when the product was released (capturing any trends if items get sold when we are predicting for volumes closer to release dates). Then we do label encoding of the categorical features so that they can be used for regression algorithms


**3.3.3. Lag and rolling lags features**\
Another important feature we observed in winning solutions is they used lags data and roll data in feature engineering. This gives how trends data could be captured using a regression algorithm , though we are not specifically using time series data.
For this we have considered rolling sum of the number of times, 0 units of product were sold, 7, 14, 30, 60, 180 days of roll (week, 2 weeks, approx month, 2 months approx, approx half year), with this we will be able to capture trend details.
As next important features we have chosen lag features (these will capture sales with a lag of that many days we have in feature.


**3.3.4. Categories - Item, Store, Department, State Level Features**\
We then use category wise sales data, item wise sales data, department wise sales data (across all stores), then also use store and category wise sales data, store and item wise sales data, store and department wise sales data. This gives cross sectional features that our model could pick if there is any trend.



**4. Model Training and Prediction**\

**Train and Predict**\
First we need to model comparison to see which model produces a better kaggle score and use that model , then optimize the step size so as to improve the score further.
through this process we are basically using the sales data that we have on t- step (for example during model selection here for 1- 14 days prediction step will be 14, and from 14 till 28th day step will be 28 days)
Then we run a prediction model where we first loop over store and department to train the model (slow) , next over store and category (will be quicker) and take average of both the methods to arrive at final submission.



**LightGBM**\
<img width="1080" alt="Screenshot 2024-01-06 at 22 29 14" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/6f93c326-e9eb-41cf-b1db-3ce0eec7ffaf">

**Extreme Gradient Boosting - XGBOOST**\
<img width="1084" alt="Screenshot 2024-01-06 at 22 30 52" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/19b07cc1-9c98-4cd2-9cd8-46d53e6599f6">


**Neural network**\
<img width="784" alt="Screenshot 2024-01-06 at 22 37 13" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/86d7c335-5570-47e0-98b1-4ec673617643">




## Result - Final Kaggle Score


| Models    | Hyperparameters                           | Kaggle Score |
|-----------|-------------------------------------------|--------------|
| LightGBM  | [See Hyperparameters](#lightgbm-parameters)| 0.5302        |
| XGBoost   | [See Hyperparameters](#xgboost-parameters)| 0.5599      |
| Neural Netwwork| [See Hyperparameters](#lightgbm-parameters)| 0.728 |



<img width="655" alt="Screenshot 2024-01-20 at 13 03 37" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/4870872b-97d6-4bbd-93d4-7039c9770134">
<img width="777" alt="Screenshot 2024-01-20 at 13 05 17" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/369ad935-bade-482b-9b3e-b8fb09b4cfd1">

## LightGBM Parameters

```python
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'min_child_weight': 1,
    'learning_rate': 0.03,
    'num_leaves': 2 ** 11 - 1,
    'min_data_in_leaf': 2 ** 12 - 1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'n_estimators': 1400,
    'boost_from_average': False,
    'verbosity': -1
}
Lgbm = LGBMRegressor(**lgb_params)
callbacks = [early_stopping(stopping_rounds=50, first_metric_only=False)]
```



## XGBoost Parameters

```python
 # Train

model = tf.keras.models.Sequential([
tf.keras.layers.Dense(64, activation='relu', input_shape=(trainX.shape[1],)),
tf.keras.layers.Dense(1, activation='linear') ]) # Linear activation for regression
                                          

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Display the model summary
#model.summary()
# Train the model
history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(valX, valY))

# Make predictions on the test set
yhat = model.predict(testX).flatten()

preds = grid[(grid['d'] >= pred_start) & (grid['d'] <= pred_end)][['id', 'd']]
preds['sales'] = yhat
predictions = pd.concat([predictions, preds], axis=0)
```



xgb_params = {
    'objective': 'reg:tweedie',  
    'eval_metric': 'rmse', 
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'learning_rate': 0.03,
    'max_depth': 11,  
    'min_child_weight': 4096,  
    'n_estimators': 1400,
    'max_bin': 100,
    'seed': 42
}








 
