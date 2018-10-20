import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def plot_missing_data(df, file_name):
    msno_plot = msno.matrix(df)
    fig = msno_plot.get_figure()
    fig.savefig('figures/'+file_name+'.png', bbox_inches='tight')


# Create categorization functions
def browser_category(item):
    '''
    INPUT: str, the feature needed to be categorized
    OUTPUT: str
    Categorize browser feature into major groups
    '''
    if item in ['Chrome','Safari','Firefox','Internet Explorer','Edge','Android Webview','Safari(in-app)']:
        return item
    elif (item == 'Opera') or (item == 'Opera Mini'):
        return 'Opera'
    else:
        return 'other'

def operatingSystem_category(item):
    '''
    INPUT: str
    OUTPUT: str
    Categorize operating system feature into major groups
    '''
    if item in ['Windows','Macintosh','Android','iOS','Linux','Chrome OS','Safari(in-app)']:
        return item
    else:
        return 'other'

def source_category(item):
    '''
    INPUT: str
    OUTPUT: str
    Categorize source feature into major groups
    '''
    if item in ['youtube.com','(direct)','dfa','baidu','siliconvalley.about.com']:
        return item
    elif item in ['google','mall.googleplex.com','analytics.google.com','google.com','sites.google.com']:
        return 'google'
    elif item in ['facebook.com','m.facebook.com','l.facebook.com']:
        return 'facebook'
    else:
        return 'other'

def revenue_category(revenue):
    if revenue != 0:
        return 1
    else:
        return 0

def add_features(df):
    df['yrMon'] = df['date'].apply(lambda x: str(x)[:6]).astype('int64')
    df['transactionRevenue'] = df['transactionRevenue'].astype('int64')
    df['isPurchase'] = df.transactionRevenue.apply(revenue_category)
    df['browserGrouping'] = df['browser'].apply(browser_category)
    df['operatingSystemGrouping'] = df['operatingSystem'].apply(operatingSystem_category)
    df['sourceGrouping'] = df['source'].apply(source_category)

def encode_feature(df):
    lb_make = LabelEncoder()
    df["channelGrouping_code"] = lb_make.fit_transform(df["channelGrouping"])
    df["browserGrouping_code"] = lb_make.fit_transform(df["browserGrouping"])
    df["operatingSystemGrouping_code"] = lb_make.fit_transform(df["operatingSystemGrouping"])
    df["deviceCategory_code"] = lb_make.fit_transform(df["deviceCategory"])
    df["continent_code"] = lb_make.fit_transform(df["continent"])
    df["sourceGrouping_code"] = lb_make.fit_transform(df["sourceGrouping"])


if __name__ == '__main__':
    df = pd.read_csv('data/raw_data.csv', index_col=0)
    plot_missing_data(df, 'before_data_clean_plot')
    # Drop unuseful columns
    df_re = df.drop(['socialEngagementType','campaign','isTrueDirect','keyword','adContent','cityId', 'latitude', 'longitude','metro', 'networkLocation', 'region','browserSize', 'browserVersion','flashVersion', 'language', 'mobileDeviceBranding','mobileDeviceInfo', 'mobileDeviceMarketingName', 'mobileDeviceModel','mobileInputSelector', 'operatingSystemVersion','screenColors', 'screenResolution'],axis=1)
    df_re = df_re.fillna(0)
    plot_missing_data(df_re, 'after_data_clean_plot')

    add_features(df_re)
    encode_feature(df_re)
    # Split train data and test data
    df_train = df_re[df_re.yrMon < 201705]
    df_test = df_re[(df_re.yrMon >= 201705) & (df_re.yrMon < 201708)]
    df_train.to_csv('data/train.csv')
    df_test.to_csv('data/test.csv')
