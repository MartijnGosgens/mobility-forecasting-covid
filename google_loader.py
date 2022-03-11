import pandas as pd
import mezuro_preprocessing as mp
import math
import copy
import datetime as dt

'''
Reduce normal mobility by a factor based on Google mobility data

Input: Mezuro mobility dataframe, historical days ago on which the prediction should be based
Output: Mezuro mobility dataframe where mobility is reduced by prediction of the Google rate

Warning: both functions take quite long to execute (preprocess: approx. 20s per date; apply: approx. 2 min. per date)
'''
#Load Google data
google_mobility=pd.read_csv(
            'data/2020_NL_Region_Mobility_Report.csv'
            ).rename(columns={'retail_and_recreation_percent_change_from_baseline': 'retail',
                         'grocery_and_pharmacy_percent_change_from_baseline': 'grocery',
                         'parks_percent_change_from_baseline': 'parks',
                         'transit_stations_percent_change_from_baseline': 'transit',
                         'workplaces_percent_change_from_baseline': 'workplaces',
                         'residential_percent_change_from_baseline': 'residential'})
#Re-format province and municipality names to match Mezuro records
google_mobility["sub_region_1"] = google_mobility["sub_region_1"].replace(
        ['North Holland','South Holland','North Brabant'],
        ['Noord-Holland','Zuid-Holland','Noord-Brabant']
        )
google_mobility["sub_region_2"] = google_mobility["sub_region_2"].replace(
    ['Government of Rotterdam','Government of Amsterdam','s-Hertogenbosch','The Hague','Eijsden','Reeuwijk','Flushing'],
    ['Rotterdam','Amsterdam',"'s-Hertogenbosch","'s-Gravenhage",'Eijsden-Margraten','Bodegraven-Reeuwijk','Vlissingen']
    )
google_mobility.loc[(google_mobility["sub_region_1"] == 'Limburg') & (google_mobility["sub_region_2"] == 'Bergen'), ("sub_region_2")] = 'Bergen (L.)'
google_mobility.loc[(google_mobility["sub_region_1"] == 'Noord-Holland')& (google_mobility["sub_region_2"] == 'Bergen'), ("sub_region_2")] = 'Bergen (NH.)'

# The gemeente naming that must be used in the outputed file
year_gemeenten_used = '2020'



def mezuro_convert_to_date(mobility,y = year_gemeenten_used):
    
    mobility = mobility.reset_index()
    mobility_overig = mobility.loc[mobility['woon'] == 'Overige gebieden']
    mobility_known = mobility.loc[mobility['woon'] != 'Overige gebieden']
    mobility_known['woon'] = mobility_known['woon'].map(lambda name: mp.koppeltabel.loc[name,'rivm'+y])
    mobility_known['bezoek'] = mobility_known['bezoek'].map(lambda name: mp.koppeltabel.loc[name,'rivm'+y])
    mobility_overig['bezoek'] = mobility_overig['bezoek'].map(lambda name: mp.koppeltabel.loc[name,'rivm'+y])
    mobility = mobility_known.append(mobility_overig)
    '''
    for index, row in mp.gemeente2gemeente.iterrows():
        print(index)
        if index[0] == 'Overige gebieden':
            mobility.rename(index={index: (index[0] ,mp.koppeltabel.loc[index[1],'rivm'+y] ,index[2]) })
        else:
            mobility.rename(index={index: (mp.koppeltabel.loc[index[0],'rivm'+y] ,mp.koppeltabel.loc[index[1],'rivm'+y] ,index[2]) })
    mobility = mobility.reset_index
    mobility = mobility[mobility['woon'] != mobility['bezoek']]
    '''
    mobility = mobility.set_index(['woon','bezoek','datum'])
    mobility = mobility.groupby(['woon','bezoek','datum']).agg({'bezoek_gemeente_id':'mean','woon_gemeente_id':'mean',  'totaal_aantal_bezoekers':'sum', 'incidentele_bezoeker':'sum', 'regelmatige_bezoeker':'sum', 'frequente_bezoeker':'sum' })
    return mobility

#mezuro_updated = mezuro_convert_to_date(mp.gemeente2gemeente)

def preprocess_rates(start_date,end_date):
    #Transform start and today date into datetime object:
    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    
    #Create empty to-be-filled dataframes
    Reduction_rates_google = pd.DataFrame(columns = ['country_region_code',	
                                                     'country_region',	
                                                     'sub_region_1',	
                                                     'sub_region_2',	
                                                     'metro_area',	
                                                     'iso_3166_2_code',	
                                                     'census_fips_code',	
                                                     'date',
                                                     'retail',	
                                                     'grocery',	
                                                     'parks',	
                                                     'transit',
                                                     'workplaces',
                                                     'residential'])
    
    Reduction_rates_google_prov = pd.DataFrame(columns = ['country_region_code',	
                                                     'country_region',	
                                                     'sub_region_1',	
                                                     'sub_region_2',	
                                                     'metro_area',	
                                                     'iso_3166_2_code',	
                                                     'census_fips_code',	
                                                     'date',
                                                     'retail',	
                                                     'grocery',	
                                                     'parks',	
                                                     'transit',
                                                     'workplaces',
                                                     'residential'])
    
    
    for t in range(1, (end_date_number - start_date_number).days + 2):
        #Calculate new date
        day_difference = dt.timedelta(days = t-1)
        current_date_number = start_date_number + day_difference
        current_date = current_date_number.strftime('%d-%m-%Y')
        current_date_google_format = current_date_number.strftime('%Y-%m-%d')
        #Extract part of Google dataframe corresponding to the current date (first only municipalities, then only provinces)
        google_mob2 = google_mobility.loc[google_mobility["sub_region_2"].notnull()].groupby('date').get_group(current_date_google_format)
        google_mob_prov = google_mobility.loc[(google_mobility["sub_region_2"].isnull()) & (google_mobility["sub_region_1"].notnull())].groupby('date').get_group(current_date_google_format)
        row_NL = google_mobility.loc[(google_mobility["sub_region_1"].isnull()) & (google_mobility["date"] == current_date_google_format) & google_mobility["sub_region_2"].isnull()]                 
        
        print(current_date)
        
        #Fill up missing values in the province data
        for index,row in google_mob_prov.iterrows():
            google_mob_prov.at[index,'date'] = current_date
            for col in ['retail','grocery','parks','transit','workplaces','residential']:
                if math.isnan(row[col]):
                    reduction_value = row_NL[col]
                    google_mob_prov.at[index,col] = reduction_value
        
        #Fill up missing values in the municipality data
        for index,row in google_mob2.iterrows():
            row_province = google_mobility.loc[(google_mobility["sub_region_1"] == row["sub_region_1"]) & (google_mobility["date"] == current_date_google_format) & google_mobility["sub_region_2"].isnull()]
            google_mob2.at[index,'date'] = current_date
            for col in ['retail','grocery','parks','transit','workplaces','residential']:
                if math.isnan(row[col]):
                    reduction_value = row_province[col]
                    if math.isnan(reduction_value):
                        reduction_value = row_NL[col]
                    google_mob2.at[index,col] = reduction_value
        
        #Add filled-up data frames
        Reduction_rates_google = Reduction_rates_google.append(google_mob2)
        Reduction_rates_google_prov = Reduction_rates_google_prov.append(google_mob_prov)
        
        #Include municipalities that do not have an entry in the Google data for the current date
        for index, row in mp.koppeltabel.iterrows():
            if len(Reduction_rates_google.loc[(Reduction_rates_google["sub_region_2"] == row["rivm"+year_gemeenten_used]) & (Reduction_rates_google["date"] == current_date)]) == 0:
                province = mp.gemeente_shapes_old.at[index,'prov_name']
                df_help_add = copy.copy(Reduction_rates_google_prov.groupby("sub_region_1").get_group(province).groupby("date").get_group(current_date))
                df_help_add["sub_region_2"] = df_help_add["sub_region_2"].fillna(row["rivm"+year_gemeenten_used])
                Reduction_rates_google = Reduction_rates_google.append(df_help_add)
    
    print(len(Reduction_rates_google))
    #Include missing municipalities from Mezuro data:
    #First: get list of municipalities in Google data:
    df_help_gemeentes = Reduction_rates_google.groupby('date').get_group(end_date)    
    for index, row in mp.koppeltabel.iterrows():
        if len(Reduction_rates_google.loc[Reduction_rates_google["sub_region_2"] == row['rivm'+year_gemeenten_used]]) == 0:
            province = mp.gemeente_shapes.at[index,'prov_name']
            df_help_add = copy.copy(Reduction_rates_google_prov.groupby("sub_region_1").get_group(province))
            df_help_add["sub_region_2"] = df_help_add["sub_region_2"].fillna(row['rivm'+year_gemeenten_used])
            Reduction_rates_google = Reduction_rates_google.append(df_help_add)
            #Note: removal of rows might not be as handy with regard to merging muniipalities
    print(len(Reduction_rates_google))            
    #Remove rows if municipality in Google data is not in rivm data:
    for index,row in df_help_gemeentes.iterrows():
        if len(mp.koppeltabel.loc[mp.koppeltabel['rivm'+year_gemeenten_used] == row["sub_region_2"]]) == 0:
            Reduction_rates_google = Reduction_rates_google[Reduction_rates_google["sub_region_2"] != row['sub_region_2']]
    print(len(Reduction_rates_google))
    return Reduction_rates_google, Reduction_rates_google_prov


    #Apply the preprocessed Google mobility rates to the Mezuro data to obtain an estimation of the actual mobility
def apply_rates(start_date,end_date,Reduction_rates_march,Reduction_rates):

    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    
    #Create new mobility file based on data from March 1-14
    
    #First: determine which days are which:
    list_day_2019 = [['04-03-2019','11-03-2019'],
                ['05-03-2019','12-03-2019'],
                ['06-03-2019','13-03-2019'],
                ['07-03-2019','14-03-2019'],
                ['01-03-2019','08-03-2019'],
                ['02-03-2019','09-03-2019'],
                ['03-03-2019','10-03-2019']
                ]
    
    list_day_2020 = [['02-03-2020','09-03-2020'],
                ['03-03-2020','10-03-2020'],
                ['04-03-2020','11-03-2020'],
                ['05-03-2020','12-03-2020'],
                ['06-03-2020','13-03-2020'],
                ['07-03-2020','14-03-2020'],
                ['01-03-2020','08-03-2020']
                ]
    
    df_mobility = pd.DataFrame(columns = ['woon',
                                   'bezoek',
                                   'datum',
                                   'bezoek_gemeente_id',
                                   'woon_gemeente_id',
                                   'totaal_aantal_bezoekers',
                                   'incidentele_bezoeker',
                                   'regelmatige_bezoeker',
                                   'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek','datum'])
    
    for t in range(0,(end_date_number - start_date_number).days + 1):
        day_difference = dt.timedelta(days = t)
        current_date_number = start_date_number + day_difference
        current_date = current_date_number.strftime('%d-%m-%Y')
        
        print(current_date)
        
        weekday = current_date_number.weekday()
        df_help_collect = pd.DataFrame(columns = ['woon',
                                       'bezoek',
                                       'datum',
                                       'bezoek_gemeente_id',
                                       'woon_gemeente_id',
                                       'totaal_aantal_bezoekers',
                                       'incidentele_bezoeker',
                                       'regelmatige_bezoeker',
                                       'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek','datum'])
        for index, i in enumerate(list_day_2019[weekday]):
            print(index,i)
            df_help_date = copy.copy(mezuro_updated.groupby("datum").get_group(i))
            
            for index2, row in df_help_date.iterrows():
                '''
                    #Make exception for overige gebieden: in that case, take mobility rates from destination gemeente - should probably be different in future iterations of the model
                df_help_date = df_help_date.reset_index()
                df_help_date_overig = df_help_date.loc[df_help_date['woon'] == 'Overige gebieden']
                df_help_date_known = df_help_date.loc[df_help_date['woon'] != 'Overige gebieden']
                df_help_date_overig['incidentele_bezoeker'] = df_help_date_overig['incidentele_bezoeker'].map(lambda value: value / (1+ float((Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == df_help_date_overig['bezoek']) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])])['retail']) / 100) * (1+float((Reduction_rates.loc[(Reduction_rates["sub_region_2"] == df_help_date_overig['bezoek']) &  (Reduction_rates["date"] == current_date)])['retail']) / 100) )
                df_help_date_overig['regelmatige_bezoeker'] = df_help_date_overig['regelmatige_bezoeker'].map(lambda value: value / (1+ float((Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == df_help_date_overig['bezoek']) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])])['retail']) / 100) * (1+float((Reduction_rates.loc[(Reduction_rates["sub_region_2"] == df_help_date_overig['bezoek']) &  (Reduction_rates["date"] == current_date)])['retail']) / 100) )
                df_help_date_overig['frequente_bezoeker'] = df_help_date_overig['frequente_bezoeker'].map(lambda value: value / (1+ float((Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == df_help_date_overig['bezoek']) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])])['workplaces']) / 100) * (1+float((Reduction_rates.loc[(Reduction_rates["sub_region_2"] == df_help_date_overig['bezoek']) &  (Reduction_rates["date"] == current_date)])['workplaces']) / 100) )
                df_help_date_known['incidentele_bezoeker'] = df_help_date_known['incidentele_bezoeker'].map(lambda value: value / (1+ float((Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == df_help_date_overig['woon']) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])])['retail']) / 100) * (1+float((Reduction_rates.loc[(Reduction_rates["sub_region_2"] == df_help_date_overig['woon']) &  (Reduction_rates["date"] == current_date)])['retail']) / 100) )
                df_help_date_known['regelmatige_bezoeker'] = df_help_date_known['regelmatige_bezoeker'].map(lambda value: value / (1+float((Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == df_help_date_overig['woon']) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])])['retail']) / 100) * (1+float((Reduction_rates.loc[(Reduction_rates["sub_region_2"] == df_help_date_overig['woon']) &  (Reduction_rates["date"] == current_date)])['retail']) / 100) )
                df_help_date_known['frequente_bezoeker'] = df_help_date_known['frequente_bezoeker'].map(lambda value: value / (1+ float((Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == df_help_date_overig['woon']) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])])['workplaces']) / 100) * (1+float((Reduction_rates.loc[(Reduction_rates["sub_region_2"] == df_help_date_overig['woon']) &  (Reduction_rates["date"] == current_date)])['workplaces']) / 100) )
               
                df_help_date = df_help_date_known.append(df_help_date_overig)
                df_help_date['totaal_aantal_bezoekers'] = df_help_date['incidentele_bezoeker'] + df_help_date['regelmatige_bezoeker'] + df_help_date['frequente_bezoeker']
                df_help_date = df_help_date.set_index(['woon','bezoek','datum'])
                
                    
                '''
                if index2[0] == 'Overige gebieden':
                    rivm_gemeente = index2[1]
                    #rivm_gemeente = mp.koppeltabel.at[index2[1],'rivm'+year_gemeenten_used]
                else:
                    #rivm_gemeente = mp.koppeltabel.at[index2[0],'rivm'+year_gemeenten_used]
                    rivm_gemeente = index2[0]
    
                rate_row = Reduction_rates_march.loc[(Reduction_rates_march["sub_region_2"] == rivm_gemeente) &  (Reduction_rates_march["date"] == list_day_2020[weekday][index])]
                rate_row2 = Reduction_rates.loc[(Reduction_rates["sub_region_2"] == rivm_gemeente) &  (Reduction_rates["date"] == current_date)]
    
                df_help_date.at[index2,'incidentele_bezoeker'] /= ((1 + float(rate_row["retail"]) / 100)  / (1 + float(rate_row2["retail"]) / 100) )
                df_help_date.at[index2,'regelmatige_bezoeker'] /= ((1 + float(rate_row["retail"]) / 100) / (1 + float(rate_row2["retail"]) / 100) )
                df_help_date.at[index2,'frequente_bezoeker'] /= ((1 + float(rate_row["workplaces"]) / 100) / (1 + float(rate_row2["workplaces"]) / 100) )
                #df_help_date.at[index2,'totaal_aantal_bezoekers'] = df_help_date.at[index2,'incidentele_bezoeker'] + df_help_date.at[index2,'regelmatige_bezoeker'] + df_help_date.at[index2,'frequente_bezoeker']
            

            df_help_collect = df_help_collect.append(df_help_date)
        
        df_help_collect = df_help_collect.groupby(['woon','bezoek']).agg({'bezoek_gemeente_id':'mean','woon_gemeente_id':'mean','totaal_aantal_bezoekers':'mean', 'incidentele_bezoeker':'mean', 'regelmatige_bezoeker':'mean', 'frequente_bezoeker':'mean' })
        df_help_collect = df_help_collect.reset_index()
        df_help_collect["datum"] = current_date
        df_help_collect = df_help_collect.set_index(['woon','bezoek','datum'])
        df_mobility = df_mobility.append(df_help_collect)
    
    df_mobility['totaal_aantal_bezoekers'] = df_mobility['incidentele_bezoeker'] + df_mobility['regelmatige_bezoeker'] + df_mobility['frequente_bezoeker']
    '''
    Mobility_df = copy.copy(df_mobility)
    '''
    '''
    #Finally, apply new reduction rates!
    for index, row in Mobility_df.iterrows():
        if index[0] == 'Overige gebieden':
            rivm_gemeente = mp.koppeltabel.at[index[1],'rivm'+year_gemeenten_used]
            print(index)
        else:
            rivm_gemeente = mp.koppeltabel.at[index[0],'rivm'+year_gemeenten_used]
        rate_row = Reduction_rates.loc[(Reduction_rates["sub_region_2"] == rivm_gemeente) &  (Reduction_rates["date"] == index[2])]
        Mobility_df.at[index,'incidentele_bezoeker'] *= (1 + rate_row["retail"] / 100) 
        Mobility_df.at[index,'regelmatige_bezoeker'] *= (1 + rate_row["retail"] / 100)
        Mobility_df.at[index,'frequente_bezoeker'] *= (1 + rate_row["workplaces"] / 100)
        Mobility_df.at[index,'totaal_aantal_bezoekers'] = Mobility_df.at[index,'incidentele_bezoeker'] + Mobility_df.at[index,'regelmatige_bezoeker'] + Mobility_df.at[index,'frequente_bezoeker']
    
    return Mobility_df
    '''
    return df_mobility

    #Create new mobility file based on data from March 1-14 without Google rates
def create_mobility_per_date(start_date,end_date):

    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    

    
    #First: determine which days are which:
    list_day_2019 = [['04-03-2019','11-03-2019'],
                ['05-03-2019','12-03-2019'],
                ['06-03-2019','13-03-2019'],
                ['07-03-2019','14-03-2019'],
                ['01-03-2019','08-03-2019'],
                ['02-03-2019','09-03-2019'],
                ['03-03-2019','10-03-2019']
                ]
    
    list_day_2020 = [['02-03-2020','09-03-2020'],
                ['03-03-2020','10-03-2020'],
                ['04-03-2020','11-03-2020'],
                ['05-03-2020','12-03-2020'],
                ['06-03-2020','13-03-2020'],
                ['07-03-2020','14-03-2020'],
                ['01-03-2020','08-03-2020']
                ]
    
    df_mobility = pd.DataFrame(columns = ['woon',
                                   'bezoek',
                                   'datum',
                                   'bezoek_gemeente_id',
                                   'woon_gemeente_id',
                                   'totaal_aantal_bezoekers',
                                   'incidentele_bezoeker',
                                   'regelmatige_bezoeker',
                                   'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek','datum'])
    
    for t in range(0,(end_date_number - start_date_number).days + 1):
        day_difference = dt.timedelta(days = t)
        current_date_number = start_date_number + day_difference
        current_date = current_date_number.strftime('%d-%m-%Y')
        print(current_date)
        weekday = current_date_number.weekday()
        df_help_collect = pd.DataFrame(columns = ['woon',
                                       'bezoek',
                                       'datum',
                                       'bezoek_gemeente_id',
                                       'woon_gemeente_id',
                                       'totaal_aantal_bezoekers',
                                       'incidentele_bezoeker',
                                       'regelmatige_bezoeker',
                                       'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek','datum'])
        for index, i in enumerate(list_day_2019[weekday]):
            df_help_date = mp.gemeente2gemeente.groupby("datum").get_group(i)
            df_help_collect = df_help_collect.append(df_help_date)
        df_help_collect = df_help_collect.groupby(['woon','bezoek']).agg({'totaal_aantal_bezoekers':'mean', 'incidentele_bezoeker':'mean', 'regelmatige_bezoeker':'mean', 'frequente_bezoeker':'mean' })
        df_help_collect = df_help_collect.reset_index()
        df_help_collect["datum"] = current_date
        df_help_collect = df_help_collect.set_index(['woon','bezoek','datum'])
        df_mobility = df_mobility.append(df_help_collect)
    
    Mobility_df = copy.copy(df_mobility)
    return Mobility_df



def create_CBS_OD_matrix():
    CBS_work_mobility=pd.read_csv(
            'data/Woon_werkafstand_werknemers__regio.csv',delimiter = ';').rename(columns={'Banen van werknemers (x 1 000)':'banen'})
    
    CBS_work_mobility["Woonregio's"] = CBS_work_mobility["Woonregio's"].replace(
    ['Groningen (gemeente)','Utrecht (gemeente)','Hengelo (O.)',"'s-Gravenhage (gemeente)",'Beek (L.)','Laren (NH.)','Middelburg (Z.)','Rijswijk (ZH.)','Stein (L.)'],
    ['Groningen','Utrecht','Hengelo',"'s-Gravenhage",'Beek','Laren','Middelburg','Rijswijk','Stein']
    )    
    CBS_work_mobility["Werkregio's"] = CBS_work_mobility["Werkregio's"].replace(
    ['Groningen (gemeente)','Utrecht (gemeente)','Hengelo (O.)',"'s-Gravenhage (gemeente)",'Beek (L.)','Laren (NH.)','Middelburg (Z.)','Rijswijk (ZH.)','Stein (L.)'],
    ['Groningen','Utrecht','Hengelo',"'s-Gravenhage",'Beek','Laren','Middelburg','Rijswijk','Stein']
    )
    CBS_work_mobility['banen'] = CBS_work_mobility['banen'].astype('float')
    CBS_work_mobility = CBS_work_mobility.set_index(["Woonregio's","Werkregio's"])
    
    df_mobility = pd.DataFrame(columns = ['woon',
                                   'bezoek',
                                   'bezoek_gemeente_id',
                                   'woon_gemeente_id',
                                   'totaal_aantal_bezoekers',
                                   'incidentele_bezoeker',
                                   'regelmatige_bezoeker',
                                   'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek'])
    
    for index, row in CBS_work_mobility.iterrows():
        if index[0] != index[1] and index[0] in mp.gemeenten_list.index and index[1] in mp.gemeenten_list.index:
            if row['banen'] != 0:
                new_row = {'woon':index[0],
                           'bezoek':index[1],
                           'bezoek_gemeente_id':mp.gemeenten_list.at[index[0],'Gemeentecode'],
                           'woon_gemeente_id':mp.gemeenten_list.at[index[1],'Gemeentecode'],
                           'totaal_aantal_bezoekers':row['banen'] * 1000,
                           'incidentele_bezoeker':0,
                           'regelmatige_bezoeker':0,
                           'frequente_bezoeker':row['banen'] * 1000
                           }
                df_mobility = df_mobility.append(new_row, ignore_index = True)
                print(index[0],index[1])
    return df_mobility



def preprocess_rates_CBS(start_date,end_date):
    #Transform start and today date into datetime object:
    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    
    #Create empty to-be-filled dataframes
    Reduction_rates_google = pd.DataFrame(columns = ['country_region_code',	
                                                     'country_region',	
                                                     'sub_region_1',	
                                                     'sub_region_2',	
                                                     'metro_area',	
                                                     'iso_3166_2_code',	
                                                     'census_fips_code',	
                                                     'date',
                                                     'retail',	
                                                     'grocery',	
                                                     'parks',	
                                                     'transit',
                                                     'workplaces',
                                                     'residential'])
    
    Reduction_rates_google_prov = pd.DataFrame(columns = ['country_region_code',	
                                                     'country_region',	
                                                     'sub_region_1',	
                                                     'sub_region_2',	
                                                     'metro_area',	
                                                     'iso_3166_2_code',	
                                                     'census_fips_code',	
                                                     'date',
                                                     'retail',	
                                                     'grocery',	
                                                     'parks',	
                                                     'transit',
                                                     'workplaces',
                                                     'residential'])
    
    
    for t in range(1, (end_date_number - start_date_number).days + 2):
        #Calculate new date
        day_difference = dt.timedelta(days = t-1)
        current_date_number = start_date_number + day_difference
        current_date = current_date_number.strftime('%d-%m-%Y')
        current_date_google_format = current_date_number.strftime('%Y-%m-%d')
        #Extract part of Google dataframe corresponding to the current date (first only municipalities, then only provinces)
        google_mob2 = google_mobility.loc[google_mobility["sub_region_2"].notnull()].groupby('date').get_group(current_date_google_format)
        google_mob_prov = google_mobility.loc[(google_mobility["sub_region_2"].isnull()) & (google_mobility["sub_region_1"].notnull())].groupby('date').get_group(current_date_google_format)
        row_NL = google_mobility.loc[(google_mobility["sub_region_1"].isnull()) & (google_mobility["date"] == current_date_google_format) & google_mobility["sub_region_2"].isnull()]                 
        
        print(current_date)
        
        #Fill up missing values in the province data
        for index,row in google_mob_prov.iterrows():
            google_mob_prov.at[index,'date'] = current_date
            for col in ['retail','grocery','parks','transit','workplaces','residential']:
                if math.isnan(row[col]):
                    reduction_value = row_NL[col]
                    google_mob_prov.at[index,col] = reduction_value
        
        #Fill up missing values in the municipality data
        for index,row in google_mob2.iterrows():
            row_province = google_mobility.loc[(google_mobility["sub_region_1"] == row["sub_region_1"]) & (google_mobility["date"] == current_date_google_format) & google_mobility["sub_region_2"].isnull()]
            google_mob2.at[index,'date'] = current_date
            for col in ['retail','grocery','parks','transit','workplaces','residential']:
                if math.isnan(row[col]):
                    reduction_value = row_province[col]
                    if math.isnan(reduction_value):
                        reduction_value = row_NL[col]
                    google_mob2.at[index,col] = reduction_value
        
        #Add filled-up data frames
        Reduction_rates_google = Reduction_rates_google.append(google_mob2)
        Reduction_rates_google_prov = Reduction_rates_google_prov.append(google_mob_prov)
        
        #Include municipalities that do not have an entry in the Google data for the current date
        for index, row in mp.koppeltabel.iterrows():
            if len(Reduction_rates_google.loc[(Reduction_rates_google["sub_region_2"] == row["rivm"+year_gemeenten_used]) & (Reduction_rates_google["date"] == current_date)]) == 0:
                province = mp.gemeente_shapes_old.at[index,'prov_name']
                df_help_add = copy.copy(Reduction_rates_google_prov.groupby("sub_region_1").get_group(province).groupby("date").get_group(current_date))
                df_help_add["sub_region_2"] = df_help_add["sub_region_2"].fillna(row["rivm"+year_gemeenten_used])
                Reduction_rates_google = Reduction_rates_google.append(df_help_add)
    
    print(len(Reduction_rates_google))
    #Include missing municipalities from Mezuro data:
    #First: get list of municipalities in Google data:
    df_help_gemeentes = Reduction_rates_google.groupby('date').get_group(end_date)    
    for index, row in mp.koppeltabel.iterrows():
        if len(Reduction_rates_google.loc[Reduction_rates_google["sub_region_2"] == row['rivm'+year_gemeenten_used]]) == 0:
            province = mp.gemeente_shapes.at[index,'prov_name']
            df_help_add = copy.copy(Reduction_rates_google_prov.groupby("sub_region_1").get_group(province))
            df_help_add["sub_region_2"] = df_help_add["sub_region_2"].fillna(row['rivm'+year_gemeenten_used])
            Reduction_rates_google = Reduction_rates_google.append(df_help_add)
            #Note: removal of rows might not be as handy with regard to merging muniipalities
    print(len(Reduction_rates_google))            
    #Remove rows if municipality in Google data is not in rivm data:
    for index,row in df_help_gemeentes.iterrows():
        if len(mp.koppeltabel.loc[mp.koppeltabel['rivm'+year_gemeenten_used] == row["sub_region_2"]]) == 0:
            Reduction_rates_google = Reduction_rates_google[Reduction_rates_google["sub_region_2"] != row['sub_region_2']]
    print(len(Reduction_rates_google))
    return Reduction_rates_google, Reduction_rates_google_prov




def apply_rates_CBS(start_date,end_date,Reduction_rates,df):
    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    
    Reduction_rates = Reduction_rates.set_index(['sub_region_2','date'])
    df_mobility = pd.DataFrame(columns = ['woon',
                                'bezoek',
                                'datum',
                                'bezoek_gemeente_id',
                                'woon_gemeente_id',
                                'totaal_aantal_bezoekers',
                                'incidentele_bezoeker',
                                'regelmatige_bezoeker',
                                'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek','datum'])
 
    
    for t in range(0,(end_date_number - start_date_number).days + 1):
        day_difference = dt.timedelta(days = t)
        current_date_number = start_date_number + day_difference
        current_date = current_date_number.strftime('%d-%m-%Y')
        print('Current date:',current_date)
        df_help_date = copy.copy(df).set_index(['woon','bezoek'])
        
        
        #Determine weekend-factor:
        Weekday_factor = 1
        '''
        
        if current_date_number.weekday() == 5:
            Weekday_factor = .36*6.2 / 31
        if current_date_number.weekday() == 6:
            Weekday_factor = .27*6.2 / 31
        if current_date_number.weekday() != 5 and  current_date_number.weekday() != 6:
            Weekday_factor = 1 - (.36+.27)*6.2 / 31
        '''
        
        for index2, row in df_help_date.iterrows():
            #print(index2)
            #Make exception for overige gebieden: in that case, take mobility rates from destination gemeente - should probably be different in future iterations of the model
            if index2[0] == 'Overige gebieden':
                rivm_gemeente = mp.koppeltabel.at[index2[1],'rivm2020']
            else:
                #rivm_gemeente = mp.koppeltabel.at[index2[0],'rivm2020']
                rivm_gemeente = index2[0]

            Target_rate_retail = Reduction_rates.at[(rivm_gemeente,current_date),'retail']
            Target_rate_workplaces = Reduction_rates.at[(rivm_gemeente,current_date),'workplaces']                
            

            
            df_help_date.at[index2,'incidentele_bezoeker']  *= (1 + Target_rate_retail / 100)  * Weekday_factor
            df_help_date.at[index2,'regelmatige_bezoeker']  *=  (1 + Target_rate_retail / 100) * Weekday_factor
            df_help_date.at[index2,'frequente_bezoeker']  *=  (1 + Target_rate_workplaces / 100) * Weekday_factor
        

        df_help_date['datum'] = current_date
        df_help_date['totaal_aantal_bezoekers'] = df_help_date['incidentele_bezoeker'] + df_help_date['regelmatige_bezoeker'] + df_help_date['frequente_bezoeker']
        #df_help_date = df_help_date.reset_index
        #df_help_date = df_help_date.set_index(['woon','bezoek','datum'])
        
        df_help_date = df_help_date.reset_index()
        
        df_mobility = df_mobility.append(df_help_date)
        
    
    return df_mobility.set_index(['woon','bezoek','datum'])
        

def create_2019_CBS(df):
    start_date = '01-03-2019'
    end_date = '14-03-2019'
    start_date_nr = dt.datetime.strptime(start_date,'%d-%m-%Y')
    end_date_nr = dt.datetime.strptime(end_date,'%d-%m-%Y')
    
    df_mobility = pd.DataFrame(columns = ['woon',
                                'bezoek',
                                'datum',
                                'bezoek_gemeente_id',
                                'woon_gemeente_id',
                                'totaal_aantal_bezoekers',
                                'incidentele_bezoeker',
                                'regelmatige_bezoeker',
                                'frequente_bezoeker'], dtype = float).set_index(['woon','bezoek','datum'])
 
    
    for t in range(0,14):
        Current_date_nr = start_date_nr + dt.timedelta(days = t)
        Current_date = Current_date_nr.strftime('%d-%m-%Y')
        
        df_HELP = copy.copy(df)
        df_HELP['datum'] = Current_date
        df_mobility = df_mobility.append(df_HELP)
    return df_mobility.set_index(['woon','bezoek','datum'])

        
    