import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Create a function to plot key metrics by significant feature
def plot_key_metrics(df, feature, figsize=(18,5)):
    '''
    INPUT:
    df: dataframe that was converted from raw_data json file
    feature: the feature that you would like to explore the key metrics for
    figsize: default is 18*5

    OUTPUT:
    This function plots the key metrics by the inputted feature.
    The key metrics are the number of visits, average revenue and the total revenue.
    '''
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    count_revenue = df.groupby(feature).count()['transactionRevenue']
    axes[0].bar(count_revenue.index, count_revenue, align='center',alpha=0.5)
    axes[0].set_title('Visits')
    axes[0].set_xticklabels(count_revenue.index,rotation=45)

    avg_revenue = df.groupby(feature).mean()['transactionRevenue']
    axes[1].bar(avg_revenue.index, avg_revenue,color = 'y', align='center',alpha=0.5)
    axes[1].set_title('Average Revenue')
    axes[1].set_xticklabels(avg_revenue.index,rotation=45)

    sum_revenue = df.groupby(feature).sum()['transactionRevenue']
    axes[2].bar(sum_revenue.index, sum_revenue,color = 'b', align='center',alpha=0.5)
    axes[2].set_title('Total Revenue')
    axes[2].set_xticklabels(sum_revenue.index,rotation=45)

    plt.savefig('figures/key_metrics_by_'+feature+'.png', bbox_inches='tight')

def plot_key_features(df, features):
    for fea in features:
        plot_key_metrics(df, fea)


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    key_features = ['channelGrouping', 'deviceCategory', 'medium', 'continent', 'browserGrouping', 'operatingSystemGrouping', 'sourceGrouping']
    plot_key_features(df, key_features)
