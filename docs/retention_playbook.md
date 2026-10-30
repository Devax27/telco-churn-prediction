# Retention playbook (illustrative)

**Status:** a method demonstration on a fictional dataset. The costs below are assumptions, not real figures, and the offers listed are hypotheses to test, not proven remedies.

## 1. Who to contact

Rule: contact a customer when the predicted churn probability is at least `offer_cost / (success_rate x value_saved)`.

With the base-case assumptions (offer 20, value of a saved customer 400, 30% chance an offer works) the threshold is 16.7%. On the held-out test set that rule contacted 53.4% of customers, reached 89.8% of the customers who left, and 44.6% of the contacted customers left. Net benefit was 17,928 per 1,000 customers against 11,852 for contacting everyone: a gain of 6,075 (95% interval 4,982 to 7,126, illustrative units).

How the rule changes with the assumptions (training-set cross-validated predictions):

| Scenario | Offer cost | Chance offer works | Threshold | Customers contacted | Precision | Recall |
|---|---|---|---|---|---|---|
| A cheap offer | 10 | 30% | 8.3% | 65.3% | 38.8% | 95.4% |
| B base case | 20 | 30% | 16.7% | 53.0% | 44.9% | 89.8% |
| C expensive offer | 50 | 30% | 41.7% | 28.2% | 60.6% | 64.4% |
| D low success rate | 20 | 10% | 50.0% | 22.3% | 65.1% | 54.6% |

In scenarios C and D, contacting everyone loses money, while a targeted list still earns a positive return.

## 2. How to prioritise when capacity is limited

Probabilities were well calibrated on the test set. Observed churn by fifth of predicted risk (lowest to highest): 1.4% (4 of 282), 7.4% (21 of 282), 20.3% (57 of 282), 37.6% (106 of 282), 66.0% (186 of 282). Work down from the highest-risk customers, using the probability as the ordering.

## 3. What is associated with churn (ideas to test)

| Pattern in the data | Observed churn | Idea to test |
|---|---|---|
| Month-to-month contract | 42.7% (one-year 11.3%, two-year 2.8%) | Incentive to move to an annual term |
| Electronic-check payment | 45.3% (other methods 15.2% to 19.1%) | Incentive to enrol in automatic payment |
| Fiber optic internet | 41.9% (DSL 19.0%) | Review fiber service quality and price |
| No online security or tech support (internet customers) | 41.8% and 41.6% (with: 14.6% and 15.2%) | Trial of a support or security bundle |
| First 6 months of tenure | 53.3% (after 49 months: 9.5%) | Early onboarding contact |

Each row is an association. Customers with these traits may differ in other ways too (fiber customers also pay more, for example), so none of these shows what an offer would do.

## 4. Pilot before scaling

The model ranks who is likely to leave. It does not tell you who would respond to an offer, and the 30% success rate is an assumption. Before committing budget:

1. Among customers above the threshold, hold out a random control group that gets no offer.
2. Compare churn between the offered and control groups over the same period.
3. Recompute the threshold with the measured success rate and real offer costs.

Break-even example: for a customer with a 45% churn probability, an offer costing 20 and worth 400 when it works pays off only if the success rate is at least 20 / (400 x 0.45) = 11.1%.

## 5. Monitoring

- Recheck calibration (predicted versus observed churn by risk group) on each new batch of customers, and retrain if they drift apart.
- Watch shifts in the customer mix (contract types, internet types), because the model learned from one snapshot.
- Audit outcomes across customer groups, which this project did not do.

## 6. Limits of this analysis

Fictional data; one snapshot without dates; tenure partly reflects who left; 374 churners in the test set, so estimates are noisy; costs and the success rate are assumed.