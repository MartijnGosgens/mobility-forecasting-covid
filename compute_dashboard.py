import constants
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    default=constants.data_dir,
    help='The directory containing the data files, e.g. the RIVM infection data and the koppeltabel.'
)
parser.add_argument(
    "--mezuro-data-dir",
    default=constants.mezuro_dir,
    help='The directory containing the Mezuro files.'
)
parser.add_argument(
    "--output-dir",
    default=constants.output_dir,
    help='The directory where the output of this script will be stored.'
)
parser.add_argument(
    "--date-yyyymmdd",
    default=None,
    help='''
        The date for which the epidemiological model is initialized, formatted as %Y%m%d. 
        We need to have RIVM infection data for at least one latent period (default 4 days) beyond this date.
    '''
)
parser.add_argument(
    "--output-gemeenten-year",
    default=2020,
    help='''
        The gemeente names from which year (2020 or 2021) should be used in the output.
    '''
)
args = parser.parse_args()
# We change the values set in the constants module. Note that this is probably bad practice.
constants.data_dir = args.data_dir
constants.mezuro_dir = args.mezuro_data_dir
constants.output_dir = args.output_dir
naming_year = args.output_gemeenten_year

from mobility_seir import MobilitySEIR,next_generation_matrix
from rivm_loader import rivm
from datetime import datetime,timedelta
import pandas as pd
from constants import CoronaConstants
from mezuro_preprocessing import to_new_gemeenten,data_path
from comparison import results_path

from estimate_transmission_rates import estimate_rates
import google_loader as gl


#Update nr of mobility
#11427249.785714285
CoronaConstants.average_total_mobility = 11427250

# Set date
mmdd = rivm.date2mmdd(datetime.today()-timedelta(days=1+CoronaConstants.latent_period))
if args.date_yyyymmdd:
    mmdd = args.date_yyyymmdd
print('We will initialize the simulation at date',rivm.mmdd2date(mmdd).strftime('%d-%m-%Y'))

date_nr = datetime.today()-timedelta(days=1+CoronaConstants.latent_period)
start_date = date_nr.strftime('%d-%m-%Y')

#Compute mobility reduction from google data
rolling_horizon = 7
end_date_rolling_google_nr = pd.to_datetime(gl.google_mobility['date']).max()
end_date_rolling_google = end_date_rolling_google_nr.strftime('%d-%m-%Y')
start_date_rolling_google_nr = end_date_rolling_google_nr - timedelta(days = rolling_horizon - 1)
start_date_rolling_google = start_date_rolling_google_nr.strftime('%d-%m-%Y')
[df_mob_march,dummy_var] = gl.preprocess_rates('01-03-2020','14-03-2020')
[df_mob_target,dummy_var] = gl.preprocess_rates(start_date_rolling_google,end_date_rolling_google)
df_mob = gl.apply_rates(start_date_rolling_google,end_date_rolling_google,df_mob_march,df_mob_target)
fraction_mobility = sum(df_mob['totaal_aantal_bezoekers']) / rolling_horizon / CoronaConstants.average_total_mobility


#Generate transmission rates (7-day average)
#Goes bakc 7 days from day before start-date
rolling_horizon_mob = 7
start_date_number_rolling = date_nr - timedelta(days = rolling_horizon)
start_date_rolling = start_date_number_rolling.strftime('%d-%m-%Y')
end_date_number_rolling = date_nr - timedelta(days = 1)
end_date_rolling = end_date_number_rolling.strftime('%d-%m-%Y')
Transmission_rates = estimate_rates(start_date_rolling,end_date_rolling)
rate_loc = sum(Transmission_rates[0]['rate_loc']) / rolling_horizon_mob
rate_mob = sum(Transmission_rates[0]['rate_mob']) / rolling_horizon_mob
frac_pos = sum(Transmission_rates[0]['frac_pos']) / rolling_horizon_mob
CoronaConstants.fraction_tested = frac_pos

Fraction_local_contacts = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl *rate_mob / rate_loc + 1)
CoronaConstants.fraction_local_contacts = Fraction_local_contacts

#Save original nr of contacts and compute new transmission probability
Contacts_average = CoronaConstants.contacts_average
CoronaConstants.contacts_average *= (1-(1-CoronaConstants.fraction_local_contacts)*(1-fraction_mobility))
transmission_prob = rate_loc / CoronaConstants.contacts_average / Fraction_local_contacts
CoronaConstants.transmission_prob = transmission_prob



gemeente2code2020 = pd.read_csv(data_path('Gemeenten{}.csv'.format(naming_year)),index_col='Gemeentenaam')['Gemeentecode']
gemeente2code2020

# Returns a matrix from target to source to number of infections
def next_generation_sources(K,infectious):
    return {
        target: (k*infectious).to_dict()
        for target, k in zip(infectious.index,K.transpose())
    }

# Translate a double dictionary indexed by 2018-gemeenten to a double
# dict indexed by 2020-gemeenten.
def double_dict_to_new_gemeenten(double_dict):
    from mezuro_preprocessing import koppeltabel
    from collections import defaultdict
    old2new = koppeltabel['rivm{}'.format(naming_year)].to_dict()
    default = lambda: defaultdict(float)
    output = defaultdict(default)
    for old_i,single_dict in double_dict.items():
        for old_j,x in single_dict.items():
            output[old2new[old_i]][old2new[old_j]] += x
    return output

def top10(ngs):
    target2top10 = {
        target: sorted(source2infections, key=source2infections.get, reverse=True)[:10]
        for target,source2infections in ngs.items()
    }
    target2top10infections = {
        target: [
            ngs[target][source]
            for source in top
        ]
        for target, top in target2top10.items()
    }
    target2top10percentage = {
        target: [
            100*x/total if total > 0 else 0
            for x in top
        ]
        for (target, top),total in zip(target2top10infections.items(),[
            sum(incoming.values())
            for incoming in ngs.values()
        ])
    }
    # Convert to dataframe
    df1 = pd.DataFrame.from_dict(target2top10,orient='index',columns=[
        'Source {}'.format(i)
        for i in range(1,11)
    ])
    df2 = pd.DataFrame.from_dict(target2top10infections,orient='index',columns=[
        'Source {} infections'.format(i)
        for i in range(1,11)
    ])
    df3 = pd.DataFrame.from_dict(target2top10percentage,orient='index',columns=[
        'Source {} percentage'.format(i)
        for i in range(1,11)
    ])
    # Collect into a single dataframe for the right order
    df = pd.DataFrame()
    for c1,c2,c3 in zip(df1.columns, df2.columns, df3.columns):
        df[c1] = df1[c1]
        df[c1+' code'] = [gemeente2code2020[a] for a in df1[c1]]
        df[c2] = df2[c2]
        df[c3] = df3[c3]
    df['target code'] = [gemeente2code2020[a] for a in df1.index]
    df.index.name = 'Target'
    return df


def top10_infection_sources(K,infectious):
    return pd.DataFrame(
        index=infectious.index,
        data=[
            (K[:,i]*infectious).sort_values(ascending=False).keys()[:10]
            for i,_ in enumerate(infectious.index)
        ],
        columns = ['Source {}'.format(i) for i in range(1,11)]
    )

from generate_figures import compute_historical_concentrations
compute_historical_concentrations(date_until=rivm.mmdd2date(mmdd))

columns = ['name','gem_id','date','predicted infections']
horizon=34



#Compute current r_eff
init_df = rivm.SEI2R2_init(mmdd)
seir_dummy = MobilitySEIR(init_df, horizon=horizon)
r_eff_estimate = seir_dummy.effective_reproduction_number_alt()
#Reset average nr of contacts
CoronaConstants.contacts_average = Contacts_average

day2date = {}
date = rivm.mmdd2date(mmdd)
for t in range(1, horizon + 1):
    date += timedelta(days=1)
    day2date[t] = date.strftime('%d-%m-%Y')

def contacts(fraction_mobility_reduction):
    return {
        "contacts_average": CoronaConstants.contacts_average * (1-(1-CoronaConstants.fraction_local_contacts)*fraction_mobility_reduction),
        "fraction_local_contacts": CoronaConstants.fraction_local_contacts /(1-(1-CoronaConstants.fraction_local_contacts)*fraction_mobility_reduction)
    }
desc2seir = {}
desc2infections = {}
init_df = rivm.SEI2R2_init(mmdd)
for r,r_desc in zip([r_eff_estimate,1.3,1.5],[str(round(r_eff_estimate,2)),'1.3','1.5']):
    for mob_red,mob_red_desc in zip([0,0.25,1-fraction_mobility],[0,0.25,str(round(1-fraction_mobility,2))]):
        seir_dummy = MobilitySEIR(init_df, horizon=horizon)
        seir_dummy.initialize_epsilon(r, r_eff_alt=True)
        seir = MobilitySEIR(init_df, horizon=horizon,constants=CoronaConstants(
            **contacts(mob_red),
            transmission_prob=seir_dummy.transmission_prob
        ))
        print(seir.constants.changed)
        seir.simulate_all()
        infections = seir.daily_reported_infections()
        data = [
            [
                a,
                gemeente2code2020[a],
                day2date[t],
                x
            ]
            for t,state in infections.items()
            for a,x in to_new_gemeenten(state,y=naming_year).items()
        ]
        df = pd.DataFrame(data=data,columns=columns)
        df.to_csv(results_path('predicted_cases_r_{}_mobility_reduction_{}.csv'.format(r_desc,mob_red_desc)),index=False)

        infectious = init_df['infected_tested'] + init_df['infected_nottested']
        K = next_generation_matrix(susceptible=init_df['susceptible'], const=seir.constants)
        top10(double_dict_to_new_gemeenten(next_generation_sources(K,infectious))).to_csv(
            results_path('predicted_top10_infection_sources_r_{}_mobility_reduction_{}.csv'.format(r_desc,mob_red_desc))
        )
        desc2seir[(r,mob_red)] = seir
        desc2infections[(r,mob_red)] = infections

