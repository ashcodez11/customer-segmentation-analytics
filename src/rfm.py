import pandas as pd

class RFMAnalyzer:
    @staticmethod
    def compute_rfm(df):
        df['RFM_Score'] = df['Recency_Score'].astype(str) + df['Frequency_Score'].astype(str) + df['Monetary_Score'].astype(str)
        df['RFM_Sum'] = df['Recency_Score'] + df['Frequency_Score'] + df['Monetary_Score']

        def rfm_segment_label(row):
            r, f, m = row['Recency_Score'], row['Frequency_Score'], row['Monetary_Score']
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 3 and f >= 1 and m >= 2:
                return 'Potential Loyalists'
            elif r == 3 and f <= 2:
                return 'Need Attention'
            elif r <= 2 and f >= 2:
                return 'At Risk'
            else:
                return 'Lost Customers'

        df['RFM_Segment'] = df.apply(rfm_segment_label, axis=1)
        return df
