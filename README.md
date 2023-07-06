# Ad Campaign A/B Test and Analysis
#### by Justin R. Papreck
https://www.kaggle.com/freshoats/ab-test-and-analysis


## Background
A large company with a substantial user base plans to increase sales using a new ad to replace their existing untargeted PSA. Prior to paying for ad space on large networks, the company wants to determine whether they will get the desired returns to make the ad deployment profitable. In order to determine this, they released the new ad and existing PSA to a sample of 20,000 users, where the PSA was the control group and the new ad as the treatment group to perform A/B testing for 31 days. 

#### Data Source
The data was sourced from a Kaggle Dataset provided by Farhad Zeynalli: https://www.kaggle.com/datasets/farhadzeynalli/online-advertising-effectiveness-study-ab-testing

The groups were separated in a 60:40 split with the 60% of the data coming from the Treatment group and 40% from the Control group. 
The columns provided represented the following: 
- Customer ID
- Test/Treatment Group
- Made Purchase
- Date with the most Ads/PSAs presented to user
- Time (Hour) of day that the the most ads were presented to user
- Total ad count seen by user

**Very little information about the meaning of these data was provided, so in order to perform a meaningful analysis, I had to make some assumptions about the data. Additionally, some patterns were discoverd in the Exploratory Analysis.**

#### Assumptions
1. There was no information provided regarding previous sales, or whether the PSA was what had been run before. I, therefore, assumed that the PSA was the existing placeholder for where the ad would appear on the platform the user was using. This allows the PSA to be a true control group. 
2. No information about the existing conversion rate. Prior to exploring the data, based on previous analyses I decided that the minimum increase in conversion would be 2%, where a 3%-4% is what the company actually hopes to achieve with this ad. (In the file, the responses were based on a sampled from the PSA subset to have a ballpark current conversion rate)
3. The author did not provide a month and year for the data collected, so when looking at the days, it was impossible to determine which days were which. Due to some pattern discrepancies, I did not assume specific dates with weekdays, but this would be easy enough to follow up with the actual company, if exists. 

---
## Analytical Preparation

We can define the control and treatment groups for the study as the PSA and ad, respectively. It's also important to define the metric that we are comparing, and here is is the conversion rate. Since the conversion rate is a quantitative measure, a T-Test is an appropriate statistical test to evaluate whether there was a significant difference between the means of the control and treatment, and since there is the potential for the treatment to have either an increased or decreased mean with respect to the control, a Two-Tailed T-Test is the more appropriate. 

H<sub>o</sub>:  Null Hypothesis 
H<sub>a</sub>:  Alternative Hypothesis
p:              Control Group Conversion Rate (PSA)
p<sub>o</sub>:  Treatment Group Conversion Rate (Ad)

**Null Hypothesis**

H<sub>o</sub>: p = p<sub>o</sub>
There will be no difference between the conversion rates of the control (PSA) and the treamtent (Ad)

**Alternative Hypothesis**

H<sub>a</sub>: p $\not=$ p<sub>o</sub>
There will be a significant difference between the conversion rates of the control (PSA) and the treament (Ad)

**Significance and Power**

$\alpha$ = 0.05
$\beta$ = 0.80

Per convention, $\alpha$ was set to 5%, giving the probability of a Type I Error (false positive). A value is considered significantly different if and only if the p-value is under $\alpha$, reducing the chance of the finding of being a false positive to under 5%. 
Also per convention, $\beta$ was set to 80%, giving the probability of a Type II Error (false Negative). This is used in determining the likelihood of testing a true effect if there is one. This will be used in the power analysis to verify that the minimum sample size required to achieve the desired significance level, effect size, and statistical power. 

---
## Experimental Preparation

**Dependencies**

```py
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
```

Typically, a Power Analysis is done prior to implementing a study and collecting survey information. The client has already implemented and collected survey data, and they have not provided us with an exact current conversion rate. What we will need to calculate for this analysis includes the effect size, statistical power, alpha, and the ratio of the number of values in the treatment group to those in the control group. For this we will go with the low end of their predicted existing conversion rate of 3% (based on a random sample of the PSA subset), and the minumum change in the range of 2%, ending with a 5% outcome. The smaller the difference between the two means, the more observations must be collected to be confident of a true difference between means not due to random sampling error.  

**Power Analysis**
 Since the data were already collected, and we do not know what parameters led the marketing team to decide on the sample size of 20,000, I decided to perform a Power Analysis to ensure that we will at least be able to make statistically significant conclusions with the standard parameters in reducing false positives and negatives. To calculate the effect size, I used the minimum improvement of 2% (as it requires more data to show a significant 2% increase than a higher percent increase). Using the sampled PSA data and the starting conversion rate of 3%, the effect size was calculated using 0.03 and 0.05 (the 2% increase from 3% to 5%). 

```py
effect_size = sms.proportion_effectsize(0.03, 0.05)
```
This yielded an effect size of -0.10286079052330155

To determine the number of observations required to meet an alpha of 0.05 and beta of 0.8, I used the NormalIndPower() function from the statmodels.stats api, which requires the effect size, power (beta), alpha, and the ratio of the data control:test.  Since the data is broken up in a 40:60 ratio, the value used for this is 1.5, since there is 1.5 times the number of control observations than treatment observations. 

```py
required_observations = sms.NormalIndPower().solve_power(
    effect_size, 
    power = 0.8, 
    alpha = 0.05, 
    ratio=1.5
)
```

The power analysis revealed that 1237 observations would be needed in each group to provide 80% confidence that there is a 2% difference between the two groups. Since the smaller of our two groups has nearly 8000 observations, we can assume that the marketing team was looking for a higher confidence level, but regardless, we do have enough data to run this test based on conventional standards. 

--- 
## Exploratory Data Analysis

The data were loaded as a dataframe with Pandas and then assessed for the following information: 
- Length of table
- Presence of duplicates
- Number of observations per test group
- Presence of Null values
- Number of dependent variable results
- Distribution of Days
- Distribution of Hours
- Distribution of Ad Count

```py
length = len(ab)
dupes = ab.duplicated().value_counts()
groups = ab['test group'].value_counts()
nulls = ab.isnull().sum().sum() purchased = ab['made_purchase'].value_counts()
```

The outputs of the above code revealed that the dataframe has 20000 rows. The number of duplicate rows is 0.
We expected to see a 40:60 ratio of PSA:Ad observations. The actual distribution is 7947 PSAs to 12053 Ads, which is not exacly a 40:60 ratio, but close.
The number of nulls across all columns and rows was 0. There were 1060 clients that purchased the product and 18940 that did not.

I used histograms to show the distribution of the other columns. The first was that of the Number of Observations by Day. This data represents the number of users that saw the maximum number of ads on that particular date, where the number 1 is the first of the month, and 30 is the thirtieth day of the month. 

![Obs_per_day](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/8170ff2d-ca0e-475b-b883-f590fb3f22e1)


This shows a uniform distribution, representing that about the same number of users saw either an ad or PSA on each day of the month, regardless of conversion. The next histogram looks at the observations during each hour of the day, on a 24 hour clock where 0 represents midnight and 23 is 11 pm. 

![Obs_per_hour](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/dcdafd52-4cb3-4bfd-8e6d-7cfb1fb707e8)


This is definitely not a uniform distribution. It shows what seems to be two different uniform distributions, one ranging from midnight to 2 pm. The second ranging from 3 pm to midnight. The earlier group has an average close to 300 users seeing their maximum ads per hour, whereas the later group is around 1700 users seeing the maximum number of ads/psas per hour. Another obervation was that there were no data recorded from 2 to 3 pm. The next histogram looks at the number of impressions by user - the total number of ads or psas seen by that user. 

![Imps_per_user](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/2f2c05c9-6c60-4686-915b-101a92a3b5b3)


This is another interesting distribution, and upon considerable analysis, this is two overapping uniform distributions, where both groups included 5 impressions. The lower number of maximum impressions was sampled with 1000 people per group, whereas the higher numbers were about 2500 per group. The values from 5 impressions reaches 3500, which is the addition of the lower sample 1000 + the larger sample 2500. 

Because these last two distributions are relatively similar in shape, with the exception of the 5 in the middle of latter, I wanted to see the numbers associated with the number of impressions and time of day. 

![Imps_per_hour](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/389cfc39-d5ef-4ed5-a18d-af3a0e51fbc0)

This shows that the maximum number of ads or psas seen is around 3 during the earlier hours of the day, and 7.5 in the afternoon and evening. This implies that it is part of the design, not coincidence. To further establish this relationship, I wanted to make sure that the numbers would add up to 20000. 

```py
early_counts = ab.loc[ab['time_with_most_impressions'] < 14]
late_counts = ab.loc[ab['time_with_most_impressions'] > 14]
print(f'There were {len(early_counts)} ads/psas run from midnight to 2 pm, and {len(late_counts)} ads/psas run from 3pm to midnight')
```

There were 5000 ads/psas run from midnight to 2 pm, and 15000 ads/psas run from 3pm to midnight.

```py
ad_counts = ab['total_user_impressions'].value_counts()
low_counts = ab.loc[ab['total_user_impressions'] < 5]
high_counts = ab.loc[ab['total_user_impressions'] > 5]
print(f'There were {len(low_counts)} ads/psas run fewer than 5 times, and {len(high_counts)} ads/psas run more than 5 times. There were {max(ad_counts)} run 5 times.')
```

There were 4001 ads/psas run fewer than 5 times, and 12576 ads/psas run more than 5 times. There were 3423 run 5 times. If 999 of the 3423 were attributed to the morning group, the remaining 2424 added to the 12576 is exactly 15000. So we can no longer treat the time of day and the maximum number of ads/psas the users were exposed to as independent events. 

---
## Data Cleaning

One of the benefits and flaws of sourcing data from Kaggle is how 'nice' the data are. This dataset is no exception. The only thing that I changed in the dataframe was the naming of the columns to be more consistent and meaningful: 

```py
new_columns = {
    'customerID': 'customer_ID', 
    'test group': 'test_group',
    'made_purchase': 'converted', 
    'days_with_most_add': 'date_with_most_impressions',
    'peak ad hours': 'time_with_most_impressions',
    'ad_count': 'total_user_impressions'}

ab.rename(columns=new_columns, inplace=True)
ab.head(3)
```

| customer_ID | test_group  | converted     | date_with_most_impressions | time_with_most_impressions | total_user_impressions |
| ---:      |    ----:      |     ---:      | ---:                       |    ----:                   |     ---:               |
| 1         | ad            | False         | 24                         | 20                         |5                       |
| 2         | psa           | False         | 21                         | 16                         |9                       |
| 3         | psa           | False         | 1                          | 18                         |8                       |         


--- 
### Statistical Analysis

What we really want to compare are the conversion rates, which we have not calculated at all yet. To do this, we need to isolate the sets of PSA and Ad in test_group, as well as the True and False in converted. In calculating the mean of the value counts per test group, the data returned will the the sum of True counts over the total number of counts per group - which gives us the conversion rate we are looking for. I wanted to create a table with the Conversion Rate, Standard Deviation, and Standard Error: 

```py
def std_p(x): return np.std(x)
def se_p(x): return stats.sem(x)
def mean(x): return np.mean(x)

conversion_rates = ab.groupby('test_group')['converted']
conversion_rates = conversion_rates.agg([mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']
conversion_rates.style.format('{:.3f}')
```

![Stats_Table](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/db7b763f-712c-4edb-8bde-1f36653d051d)


![Stats_graph](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/31973519-1ecf-4131-be30-d020e5c1d6a7)


The error bars on the graph represent the standard error. While these groups look significantly different, favoring the conversion by the ads over the PSAs, I subsequently ran a Z-Test to verify this. Why a Z-Test and not a T-Test? Because with the number of observations greater than 100, the T and Z distributions are nearly identical. 

```py
# Separate the results sets
control_results = ab[ab['test_group'] == 'psa']['converted'] 
treatment_results = ab[ab['test_group'] == 'ad']['converted']

# Count subtotals
n_control = control_results.count()
n_treatment = treatment_results.count()
conversions = [control_results.sum(), treatment_results.sum()]
n_observations = [n_control, n_treatment]

z_stat, pval = proportions_ztest(conversions, nobs = n_observations) # Proportion Z-Test
(lower_control, lower_treatment), (upper_control, upper_treatment) = proportion_confint(conversions, nobs=n_observations, alpha = 0.05) # Proportion Confidence Interval

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.28f}')
print(f'ci 95% for control group: [{lower_control:.3f}, {upper_control:.3f}]')
print(f'ci 95% for treatment group: [{lower_treatment:.3f}, {upper_treatment:.3f}]')
```
z statistic: -10.59
p-value: 0.0000000000000000000000000331
ci 95% for control group: [0.028, 0.036]
ci 95% for treatment group: [0.062, 0.071]

Note: I increased the number of decimals for the p-value to show how much lower than 0.05 the value was determined to be. 

```py
minimum_difference = (lower_treatment - upper_control)*100
maximum_difference = (upper_treatment - lower_control)*100
psa_mean = conversions[0]/n_control
ad_mean = conversions[1]/n_treatment
mean_difference = (ad_mean-psa_mean)*100
```

The range of percent increase with the ad spans from the minumum of 2.59% to a maximum of 4.26% within the confidence interval of 80%.
The difference between the means was 3.43%.
The conversion rate from the psa on the landing page was 3.23%, so with 80% confidence, the true mean of the psa conversion rate will be between 2.84% and 3.62%.
The conversion rate from the ad on the landing page was 6.66%, so with 80% confidence, the true mean of the psa conversion rate will be between 6.22% and 7.11%.
The p-value was substantially lower than 0.05, on the order of 10^-26, thus indicating that the difference in means of the control and treatment was significant.


### Statistical Results

The p-value was found to be 3.31*10<sup>-26</sup>, which definitely falls below the 5% threshold for meeting significance. What the p-value tells us is only that the two groups are indeed different, and in this case have a very low chance of yielding false positives. The confidence interval for the psa group was 2.84% to 3.62%, denoting that we are 80% confident that the true mean of the population of all users reaching the landing page with the psa falls between those values. The mean conversion rate for the psa group was 3.23%. This finding supports the idea that the psa on the landing page is in the same range as the provided current conversion rate that the company reported, between 3 and 4 percent, though the estimate may have been a bit ambitious. The confidence interval for the new ad campaign was 6.22% to 7.11%, with a mean of 6.66% converted. 

The company initially expressed that they were hoping to bump up conversions to somewhere around 7-8%, but they would accept anything greater than a 2% increase. These findings suggest that the goal of 7% is not out of reach with this campaign based on the sampled data. Looking at the range of increases, the increase in conversion rate ranged from 2.59% to 4.26%. Based on the significant difference established by the p-value and the differences established using the 80% confidence intervals, the minumum increase exceeds the 2% threshold presented by the stakeholder in deciding whether to run the campaign. These findings suggest that they will likely see an increase in sales closer to 3.4%. 

---
## Additional Insights

Now that we know that the customer will most likely run the ad, if they decide to run the ad externally they would like to know when to run the ads and if there is a correlation between the number of ads that users see before leading to conversion. For this I needed to actually separate the groups by ad and PSA, then whether conversion was true: 

```py
# Separate the two groups
ads = ab[ab['test_group']=='ad']
psas = ab[ab['test_group']=='psa']

# Total Conversions per group
ads_converted = ads[ads['converted']==True]
psas_converted = psas[psas['converted']==True]
```

The first question I'd like to address is **How many ads to run per hour?**

![Ad_imps_converted](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/a33cb3dc-f089-422b-b13e-1d58b36966c1)

The above figure shows the number of maximum ads run and their associated conversion rates. Because the number of maximum ads run is dependent on the time of day, it isn't surprising to see this is a bimodal distribution, with maxima at both 2 ads and 7 ads. Since the goal for the company is the run as few ads as possible to optimize profitability, running more than 7 ads during the afternoon/evening hours seems excessive without gain. 

![PSA_imps_converted](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/d2005ff0-6c99-4f84-8e72-54457ff53369)


Unlike the Ad conversion distribution, it doesn't seem like there is a clear maximum in the early group, and for the later group the highest number of psas seen yielded the highest conversion rate. Even the highest conversion rate at 10 impressions per hour only yielded a 4% conversion rate, lower than the lowest conversion rate of the ads run.  

The next question to address is **When should the ads be run?**

In the exploratory analysis, it was clearly shown that there were a lower number of users from midnight to 2 pm as well as a lower number of ads or psas shown to those users during those hours, from 1 to 5, whereas the later group had 5 to 10 ads, and many more users saw such ads and psas. However, the conversion rates somewhat normalize this since it is the mean count of conversions per total. The Conversion Rates running ads and psas are shown below, respectively. 

![Ad_imps_per_hour](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/540ad9d5-9bf1-48c8-9e9f-0c7502835f04)

![psa_imps_per_hour](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/257c1e3f-50a2-49a6-b0a5-25a185fc37bc)


The first thing to note is that there is no hour 14; this is the hour when no ads or psas were run. Interestingly, some of the highest conversion rates for both the PSA and the Ad was from 5 am to 6 am. Considering the standard errors shown by the error bars, it is clear that the SE for the earlier times is substantially greater than those higher than 14, and this is an artifact of the quantity of data collected for each of these sets. 

There seem to be four periods of the day in the Ad data that have elevated: 
- 0 to 3
- 5 to 8 
- 12 to 13
- 18 to 20

Much of this pattern is reflected in the PSA data, that have elevated levels in similar time periods:
- 0 to 2
- 5 to 6 and 7 to 9
- 11 to 12
- 21 to 22

What this tells us is that there is an overall online presence during these time chunks, and these would be ideal for running the ads. At all times throughout the day, the ad demonstrated a higher conversion rate than did the PSA. The worst times of day to run the ads would be from 2 to 5, 10 to 12, and 13 to 15. 

The final question is **What days should the ads be run?**

![Ad_imps_by_date](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/55e68439-f420-4ffa-817a-4ef6de52f67e)


What can be seen with the ad data is that there is a clear pattern when at the start of the week* the conversion rate is lowest, and then it increases throughout the rest of the week before dropping again. This isn't a perfect pattern, and more problematic is that there is no information about which weekdays line up with the dates. We also have no idea what month or year these data come from. It would make a huge difference if this were representing July vs. December, which could explain the last week breaking the pattern. Unfortunately, without that specific information, I can't make a suggestion on which weekdays to focus on, however if these data were presented, I could inform the company to push more ads out on the particular days of the week that yield the highest conversions. The month also makes a big difference because users are more likely to be indoors and home during the winter vs. the summer. So while there is a bit of a pattern here, there isn't enough information to provide a good recommendation. 

![psa_imps_by_date](https://github.com/FreshOats/Marketing_Analysis/assets/33167541/03fd774b-8445-4f2c-b5c1-9c1bc22fc023)

A similar pattern is found here with the PSAs as with the Ad campaign, suggesting that there are definitely better days for conversion during the week, but more information is needed to determine which days. What this also indicates is that regardless of the media presented to the users, the conversion rates will be less dependent on the ad itself and more dependent on the day the potential customer decides to make a purchase. 

These results further support the hourly results, that there are specific times and days of the week that have higher potential for consumers to make online purchases. The reason that knowing the month and year are so important is to determine why there were such spikes in PSA purchases on the 4th, 14th, and 18th. In fact, on the 4th, the PSA conversion rate cannot be considered different from the Ad conversion rate, and likewise on the 18th, where according to these data the conversion rate of the PSA was higher than Ad conversion rate. One suspicion I have is that this represents the month of December - the is a huge increase up to the 23rd, but then a decline on the 24th through 26th, but then a huge jump a few days later. This is the only week to really break pattern from the increase/sharp decline throughout the rest of the month. 


---

# Summary
1. **Should the company run the ad campaign?**

    The results show that there was a 6.7% conversion rate while running the ad compared to the 3.2% conversion rate while maintaining the PSA. Using a Z-Test, the p-value criterion for significance was met, so the Null Hypothesis was rejected. Thus, there was a significant difference between the Ad and the PSA. 

    The client had a goal of a 7-8% conversion rate, or a 3-4% increase in conversion. They decided the lowest permissible increase would be 2% to abandon this ad campaign. In considering the confidence interval, we can state that we are 95% confident that the true mean of the ad campaign when run will be between 6.22% and 7.11%, meeting the goal of the client, with 7%. The Sample Mean conversion rate was 6.66%. 

    Given this information, I would advise the client to run the ad campaign. 


2. **When should they run the ads?**

    In analyzing the conversion rates for both the PSA and Ads, there were times of day when conversion rates were lower than other times of day, and prove to be the least effective times to run ads. These times were from 1 am to 5 am, 9 am to noon, and 1 pm to 3 pm. 

    The ads were most effective from 5 am to 8 am, noon, and then 4 pm to 10 pm, with the highest conversion rates from 5 am to 6 am and 7 am to 8 am. The ads run between 4 pm and 10 pm were fairly consistent in range, peaking at 6 pm. 


3. **How many ads should they run per hour?**
 
    The conversion rates were the highest when users saw 2 ads in the morning/early afternoon and 7 ads in the late afternoon/evening, reaching around 8% conversion. That being said, there is no data in the afternoon of whether showing 2 ads per hour would remain as effective as is in the morning, and could potentially reduce the cost of running ads at that time. It is also important to consider running 7 ads per hour in the morning, especially with the 5 am to 8 am group - this may increase the conversion rate even higher. But with the given sample data, my recommendation is to run 2 per hour in the morning, and 7 per hour later in the day.


## Followup
Information that would have been useful in parts of this analysis include:

**Full dates for the sampled data**
As mentioned numerous times before, the time of week and time of year are more important than time of month, unless we're considering lunar cycles, which could be influencial for coastal products and other fauna and flora impacted by lunar cycles. That being said, we still need the dates to determine the lunar cycling.

**Type of products that are sold by this company**
Does this company sell products that are designed for adults, teens or children? Based on the data, it would not be implausible that they sell coffee or breakfast foods or perhaps gym equipment. This would explain the high morning conversion on both the Ad and PSA. Knowing the industry that the company represents will help identify true artifacts versus actual patterns.  

**Cost range of the products purchased**
The binary conversion measurement may not be profitable if most of the products purchased are low-cost, low-profit items. Selling breakfast bars is not going to have the same impact as selling high-end coffee/espresso makers. 

