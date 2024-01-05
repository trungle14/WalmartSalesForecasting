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
sales_train_validation.csv	historical daily unit sales data per product and store [d_1 - d_1913]	1019	30.5 K	120 MB
sales_train_evaluation.csv	sales [d_1 - d_1941]	1047	30.5 K	121.7 MB
Based on the structure of data we see the data would be of below format:

<img width="685" alt="Screenshot 2024-01-04 at 20 00 36" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/d5acb20b-a021-4245-8f3d-27e581c4b1a9">


<img width="488" alt="image" src="https://github.com/trungle14/WalmartSalesForecasting/assets/143222481/90b7b915-c581-4a11-85f5-111f6f50411d">





 
