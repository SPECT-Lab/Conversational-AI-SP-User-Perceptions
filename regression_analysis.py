import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.api as sm
from datetime import timedelta
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
import sys

if len(sys.argv) < 2:
    raise ValueError("No CSV file provided. Please run the script with a CSV file as an argument. Example: python train_model.py <path_to_your_csv_file>")

posts_df = pd.read_csv(sys.argv[1])
posts_df = posts_df.fillna("")

posts_df['Date'] = pd.to_datetime(posts_df['Date'])
posts_df['date'] = pd.to_datetime(posts_df['Date'])

themes_subthemes = {
    'Data Collection': ['Personal data', 'Proprietary information'],
    'Data Usage': ['Model training', 'Third-party sharing'],
    'Data Retention': ['Data retention'],
    'Security Vulnerabilities': ['Platform security', 'LLM-powered applications', 'LLM-generated code'],
    'Legal Compliance': ['GDPR', 'HIPAA'],
    'Transparency and Control': ['Transparency and control']
}
themes = list(themes_subthemes.keys())

events = [
    {"title": "Launch of ChatGPT", "date": "2022-11-30"},
    {"title": "Microsoft’s acquisition of OpenAI", "date": "2023-01-01"},
    {"title": "Introduction of ChatGPT Plus", "date": "2023-02-01"},
    {"title": "ChatGPT API becomes available", "date": "2023-03-01"},
    {"title": "Samsung employees leak sensitive data to ChatGPT", "date": "2023-04-04"},
    {"title": "Italy temporarily bans ChatGPT", "date": "2023-04-01"},
    {"title": "Launch of GPT-4", "date": "2023-03-14"},
    {"title": "Security bug: Mixing of chat histories", "date": "2023-03-20"},
    {"title": "Release of Bard", "date": "2023-03-21"},
    {"title": "ChatGPT plugins", "date": "2023-03-23"},
    {"title": "Ability to turn off history introduced in ChatGPT", "date": "2023-04-25"},
    {"title": "The White House statement on principles and development of AI", "date": "2023-05-01"},
    {"title": "Updated privacy policy", "date": "2023-06-23"},
    {"title": "FTC investigation into OpenAI", "date": "2023-07-01"},
    {"title": "Introduction of code interpreter", "date": "2023-07-06"},
    {"title": "Llama 2 released", "date": "2023-07-18"},
    {"title": "Custom Instructions for ChatGPT", "date": "2023-07-20"},
    {"title": "GPT-3.5 turbo fine-tuning", "date": "2023-08-22"},
    {"title": "Release of GPT Enterprise", "date": "2023-08-28"},
    {"title": "Poland opens privacy probe of ChatGPT", "date": "2023-09-21"},
    {"title": "ChatGPT can now see, hear, and speak", "date": "2023-09-25"},
    {"title": "Copyright lawsuits against LLM developers", "date": "2023-10-01"},
    {"title": "DALL·E 3 in ChatGPT Plus and Enterprise", "date": "2023-10-19"},
    {"title": "OpenAI Dev Day", "date": "2023-11-07"},
    {"title": "Release of GPTs", "date": "2023-11-06"},
    {"title": "Updated privacy policy", "date": "2023-11-14"},
    {"title": "OpenAI announces leadership transition", "date": "2023-11-17"},
    {"title": "Sam Altman returns as CEO", "date": "2023-11-29"},
    {"title": "Gemini released", "date": "2023-12-06"},
    {"title": "Europe Privacy Policy", "date": "2023-12-15"},
    {"title": "Release of ChatGPT Team", "date": "2024-01-10"},
    {"title": "Release of GPT store", "date": "2024-01-10"},
    {"title": "Release of ChatGPT memory feature", "date": "2024-02-12"},
    {"title": "EU AI Act passed", "date": "2024-03-12"},
    {"title": "Llama 3 released", "date": "2024-04-18"},
    {"title": "GPT-4o", "date": "2024-05-13"},
    {"title": "Scarlett Johansson’s copyright allegation against OpenAI", "date": "2024-05-21"}
]

def check_autocorrelation(residuals):
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    p_value = lb_test['lb_pvalue'].values[0]
    return p_value < 0.05 


def itsa_event_analysis(data, event_date, event_name, pre_period=7, post_period=7):
    
    start_date = event_date - timedelta(weeks=pre_period)
    end_date = event_date + timedelta(weeks=post_period)
    model_data = data.loc[start_date:end_date].copy()
    
    if len(model_data) < (pre_period + post_period):
        print(f"Not enough data for event on {event_date}")
        return None, None
    
    model_data.reset_index(inplace=True)
    
    model_data['time'] = np.arange(len(model_data))
    
    model_data['intervention'] = (model_data['date'] >= event_date).astype(int)
    
    model_data['time_after_intervention'] = model_data['time'] * model_data['intervention']
    
    X = model_data[['time', 'intervention', 'time_after_intervention']]
    
    X = sm.add_constant(X) 
    y = model_data['daily_count']
    
    model = sm.OLS(y, X).fit()
    
    if check_autocorrelation(model.resid):
        print(f"Autocorrelation detected for event on {event_date}. {event_name}")
    else:
        print(f"No autocorrelation detected for event on {event_name}.")
    
    return model, model_data

for theme, subthemes in themes_subthemes.items():
    for subtheme in subthemes:
        print("\n\n", subtheme)
        daily_counts = posts_df[posts_df[subtheme] == 'Yes'].resample('D', on='date').size().reset_index(name='daily_count')

        reddit_data = daily_counts
        events_data = pd.DataFrame(events)
        
        reddit_data['date'] = pd.to_datetime(reddit_data['date'])
        events_data['date'] = pd.to_datetime(events_data['date'])
        
        reddit_data.set_index('date', inplace=True)
        reddit_data = reddit_data.asfreq('D')
        
        reddit_data['daily_count'] = reddit_data['daily_count'].fillna(method='ffill')
        
        results_list = []
        
        for index, row in events_data.iterrows():
            event_date = row['date']
            event_name = row['title']
            
            model, model_data = itsa_event_analysis(reddit_data, event_date, event_name)
            
            if model is not None:
                p_values = model.pvalues
                coefficients = model.params
                results_list.append({
                    'event_name': event_name,
                    'event_date': event_date,
                    'intervention_coef': coefficients['intervention'],
                    'intervention_p_value': p_values['intervention'],
                    'trend_change_coef': coefficients['time_after_intervention'],
                    'trend_change_p_value': p_values['time_after_intervention']
                })
            else:
                print(f"Skipping event '{event_name}' due to insufficient data.")


        results_df = pd.DataFrame(results_list)

        if not results_df.empty:
            p_values = np.hstack([results_df['intervention_p_value'], results_df['trend_change_p_value']])
        
            adjusted_results = multipletests(p_values, alpha=0.05, method='bonferroni')
            adjusted_p_values = adjusted_results[1]
        
            num_events = len(results_df)
            results_df['adj_intervention_p_value'] = adjusted_p_values[:num_events]
            results_df['adj_trend_change_p_value'] = adjusted_p_values[num_events:]
        
            results_df['intervention_significant'] = results_df['adj_intervention_p_value'] < 0.05
            results_df['trend_change_significant'] = results_df['adj_trend_change_p_value'] < 0.05
            results_df['is_significant'] = results_df['intervention_significant'] | results_df['trend_change_significant']
        
            significant_events = results_df[results_df['is_significant']]
        
            print("\nSignificant Events After Correction:")
            print(significant_events[['event_name', 'event_date', 'intervention_coef', 'adj_intervention_p_value', 'trend_change_coef', 'adj_trend_change_p_value']])
        else:
            print("No results to display.")