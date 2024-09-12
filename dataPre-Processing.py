from ast import Str
from configparser import Interpolation
from operator import index
from unicodedata import numeric
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import datetime
import time
#PROGRAMMA PRINCIPALE MAIN #kettle
directory_ukdale = os.path.join("C:\\", "Users","silve","OneDrive","Desktop", "UKDALE")
start_time = time.time()
sample_seconds = 6
parametri_apparati = {
    'kettle': {
        'windowlength': 600,
        'houses': [1, 2, 5], #ok
        'channels': [10, 18, 2, 18], #ok
        'train_build': [1, 5], #ok
        'test_build': [2], #ok
        'step_size': 16  #ok
    },
    'microwave': {
        'windowlength': 600,      
        'houses': [1, 2,5],
        'channels': [13, 15, 23],
        'train_build': [1,5],
        'test_build': [2],
        'step_size': 24
    },
    'fridge': {
        'windowlength': 600,
        'houses': [1, 2,5],
        'channels': [12, 14,5,19],
        'train_build': [1,5],
        'test_build': [2],
        'step_size': 28
    },
    'dishwasher': {
        'windowlength': 600,
        'houses': [1, 2, 5],
        'channels': [13, 22, 13], #6
        'train_build': [1,5],
        'test_build': [2],
        'step_size': 32
    },
    'washingmachine': {
        'windowlength': 600,
        'houses': [1, 2, 5],
        'channels': [5, 12, 24],
        'train_build': [1,5],
        'test_build': [2],
        'step_size': 8
    }
}
#salvataggio file csv 
def savefile(df1,title):
     print(df1)
     df1.to_csv(os.path.join("C:\\", "Users","silve","OneDrive","Desktop",str(title)+".csv"))
     print("file salvato correttamente")


# read flash.dat to a list of lists
def load_file(dir,col_names=['time','data'], nrows=None):
    df= pd.read_table( dir,
                       sep="\s+",
                       nrows=None,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df


def createdatacsv():   
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df_aggregate=pd.DataFrame()
    df_appliance=pd.DataFrame()
    df_appliance2=pd.DataFrame()
    df_merged=pd.DataFrame()
    for h in parametri_apparati[appliance_name]['train_build']:       
        #faccio print directory riferito alla singola casa e al canale dell'apparato
            print(directory_ukdale + "\\"+ 'house_' + str(h) + "\\"
                  + 'channel_' +
                 str(parametri_apparati[appliance_name]['channels'][parametri_apparati[appliance_name]['train_build'].index(h)]) +
                  '.dat')
            directory_appliance= (directory_ukdale + "\\"+ 'house_' + str(h) + "\\"
                  + 'channel_' +
                 str(parametri_apparati[appliance_name]['channels'][parametri_apparati[appliance_name]['train_build'].index(h)]) +
                  '.dat')
            
            directory_aggregate= (directory_ukdale + "\\"+ 'house_' + str(h) + "\\"
                  + 'channel_1.dat')
            print(directory_aggregate)
      #carico dataframe singolo apparato 
            if(df_appliance.empty):
                print(directory_appliance)
                df_appliance=load_file(directory_appliance)
                df_appliance['time']=pd.to_datetime(df_appliance['time'], unit='s')
                #df.set_index('time',inplace=True)
                
                print("df:")
                print(df_appliance.head())
            
                plt.plot(df_appliance['time'], df_appliance['data'])
                plt.show()

                print(directory_aggregate)
                df_aggregate=load_file(directory_aggregate)
                df_aggregate['time']=pd.to_datetime(df_aggregate['time'], unit='s')
                print("df_aggregate:")
                print(df_aggregate.head())
                #df = df_aggregate.merge(df_appliance, on='time', suffixes=('_df1', '_df2'))

                # the timestamps of mains and appliance are not the same, we need to align them
                # 1. join the aggragte and appliance dataframes;
                # 2. interpolate the missing values;
                df_aggregate = df_aggregate.add_suffix('_df_aggregate')
                df_appliance = df_appliance.add_suffix('_df_appliance')

                df_aggregate.set_index('time_df_aggregate', inplace=True)
                df_appliance.set_index('time_df_appliance', inplace=True)

                df_align = df_aggregate.join(df_appliance, how='outer'). \
                    resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
                df_align = df_align.dropna()

                df_align.reset_index(inplace=True)

                print('Numero di righe prima casa ' , len(df_align))
                #df = df_aggregate.join(df_appliance, how='outer').fillna(method='backfill', limit=1)
                #print("df_main")
                #print(df.head())
            else:
                
                print(directory_appliance)
                df_appliance2=load_file(directory_appliance)
                df_appliance2['time']=pd.to_datetime(df_appliance2['time'], unit='s')
                #df.set_index('time',inplace=True)
                
                #df_merged = df_appliance.append(df_appliance2, ignore_index=True)

            
                print("df:")
                print(df_appliance2.head())
            
                plt.plot(df_appliance2['time'], df_appliance2['data'])
                plt.show()
                #print("df_merged:")
                #print(df_merged.head())
            
                #plt.plot(df_merged['time'], df_merged['data'])
                #plt.show()


      #carico l'aggregato
            
            #if(df_aggregate.empty):
                
                print(directory_aggregate)
                df_aggregate2=load_file(directory_aggregate)
                df_aggregate2['time']=pd.to_datetime(df_aggregate2['time'], unit='s')
                #df_mergedaggregate = df_aggregate.append(df_aggregate2, ignore_index=True)
                #print("df_mergedaggregate:")
                #print(df_mergedaggregate.head())
                #df2 = df_aggregate.merge(df_merged, on='time', suffixes=('_df1', '_df2'))
                #print(df2.head)
                #df = df_aggregate.join(df_appliance, how='outer').fillna(method='backfill', limit=1)
                
                #plt.plot(df_aggregate['time'], df_aggregate['data'])
                #plt.show()

                
                 # the timestamps of mains and appliance are not the same, we need to align them
                # 1. join the aggragte and appliance dataframes;
                # 2. interpolate the missing values;
                df_aggregate2 = df_aggregate2.add_suffix('_df_aggregate')
                df_appliance2 = df_appliance2.add_suffix('_df_appliance')

                df_aggregate2.set_index('time_df_aggregate', inplace=True)
                df_appliance2.set_index('time_df_appliance', inplace=True)

                df_align2 = df_aggregate2.join(df_appliance2, how='outer'). \
                    resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
                df_align2 = df_align2.dropna()
                
                df_align2.reset_index(inplace=True)
                print('Numero di righe seconda casa ' , len(df_align2))
                #df_mergedaggregate = df_mergedaggregate.add_suffix('_df_aggregate')
                #df_merged = df_merged.add_suffix('_df_merged')
                # the timestamps of mains and appliance are not the same, we need to align them
                # 1. join the aggragte and appliance dataframes;
                # 2. interpolate the missing values;
                #df_mergedaggregate.set_index('time_df_aggregate', inplace=True)
                #df_merged.set_index('time_df_merged', inplace=True)

                #df_align = df_mergedaggregate.join(df_merged, how='outer'). \
                #resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
                #df_align = df_align.dropna()

               # df_align.reset_index(inplace=True)
                
                merged_dataset = pd.concat([df_align, df_align2], ignore_index=True)
                #df_align = df_mergedaggregate.join(df_merged, how='outer')
                #df_align = df_align.resample('6S').mean()
                #df_align = df_align.ffill()
                #df_align = df_mergedaggregate.join(df_merged, how='inner'). \
                #    resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
                #df_align = df_align.dropna()

                #df_align.reset_index(inplace=True)
                #df= df1.append(df2, ignore_index=True)
                print("df_main")
                print(df_align.head())
                print(df_align)
                num_rows = merged_dataset.shape[0]
                print("Il dataframe merged_dataset ha", num_rows, "righe.")

            
    
                
    return merged_dataset   
    #return df_align
# Rimozione dei dati mancanti (missing data) dal dataset
def remove_missing_data(data):
    data = data.dropna()
    return data

# Divisione delle sequenze in sotto-sequenze quando la durata � inferiore a 20 secondi
def split_subsequences(data, time_column, threshold):
    subsequences = []
    subseq = []
    #range(len(data)):
    for i in range(len(data)):
        subseq.append(data.iloc[i])
        if (i < len(data) - 1) and ((data['time_df_merged'].iloc[i+1] - data['time_df_merged'].iloc[i]).seconds >= threshold):
            subsequences.append(subseq)
            subseq = []
    subsequences.append(subseq)
    return subsequences

# Rimozione delle sequenze con y(t) < 10 watts per bilanciare lo stato on/off
#def balance_samples(subsequences, target_column, threshold):
#    balanced_subsequences = []
#    for subseq in subsequences:
#        y = [sample['data_df_merged'] for sample in subseq]
#        if sum(y) >= threshold:
#            balanced_subsequences.append(subseq)
#    return balanced_subsequences


def balance_samples(x, y, threshold=10, ratio=1):
    # Trova gli indici dei campioni dove y(t) >= threshold
    active_indices = np.where(y >= threshold)[0]

    # Trova gli indici dei campioni dove y(t) < threshold
    inactive_indices = np.where(y < threshold)[0]

    # Determina il numero di campioni attivi e inattivi
    num_active_samples = len(active_indices)
    num_inactive_samples = len(inactive_indices)

    # Calcola il numero di campioni da selezionare per ogni classe per bilanciare il rapporto
    num_samples_to_select = min(num_active_samples, num_inactive_samples * ratio)

    # Seleziona casualmente i campioni da entrambe le classi
    selected_active_indices = np.random.choice(active_indices, num_samples_to_select, replace=False)
    selected_inactive_indices = np.random.choice(inactive_indices, num_samples_to_select, replace=False)

    # Concatena gli indici selezionati
    selected_indices = np.concatenate([selected_active_indices, selected_inactive_indices])

    # Ordina gli indici selezionati per mantenere l'ordine originale
    selected_indices.sort()

    # Seleziona i dati corrispondenti agli indici selezionati
    selected_x = x[selected_indices]
    selected_y = y[selected_indices]

    selected_x=selected_x.reset_index(drop=True)
    selected_y = selected_y.reset_index(drop=True)

    return selected_x, selected_y

# Normalizzazione dei valori di input
def normalize_data(data):
    max_value = max(data)
    normalized_data = data / max_value
    return normalized_data



###
def gap_over(data):
    x=0
    x=len(data)
    for i in range(1000000):       
        if(data['index'].iloc[i+1] - data['index'].iloc[i]>=  datetime.timedelta(minutes=2) ):
        #aggiuno 0 ogni 6 secondi in base al campione di UK-DALE 
            y=i
            for j in range(20):
                data['data_df_merged'][y] = 0
                data['data_df_aggregate'][y] = 0
                #data.insert(loc=y, column='data_df2', value=0)
                #data.insert(loc=y, column='data_df1', value=0)           
                y=+1
    return data

def interpolation(data):
    data=pd.DataFrame(data)
    data.fillna(0, inplace=True)
   
    return data


# Configurazione del blocco residuo
      
      #residual = tf.keras.layers.Conv1D(1,1)(input_data)
      #residual= input_data

      #x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="causal", dilation_rate=2, activation='relu')(residual)
      #x = tf.keras.layers.BatchNormalization()(x)
      #x = tf.keras.layers.Dropout(rate=0.1)(x)
      #x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="causal", dilation_rate=2, activation='relu')(x)
      #x = tf.keras.layers.BatchNormalization()(x)
      #x = tf.keras.layers.Dropout(rate=0.1)(x)
      #x = tf.keras.layers.Add()([residual,x])

      #residual_2 = tf.keras.layers.Conv1D(1,1)(x)

      #x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", dilation_rate=4, activation='relu')(residual_2)
      #x = tf.keras.layers.BatchNormalization()(x)
      #x = tf.keras.layers.Dropout(rate=0.1)(x)
      #x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal",dilation_rate=4, activation='relu')(x)
      #x = tf.keras.layers.BatchNormalization()(x)
      #x = tf.keras.layers.Dropout(rate=0.1)(x)
      #x = tf.keras.layers.Add()([residual_2,x])

      #residual_3 = tf.keras.layers.Conv1D(1,1)(x)

      #x = tf.keras.layers.Conv1D(filters=64, kernel_size=3,padding="causal", dilation_rate=8, activation='relu')(residual_3)
      #x = tf.keras.layers.BatchNormalization()(x)
      #x = tf.keras.layers.Dropout(rate=0.1)(x)
      #x = tf.keras.layers.Conv1D(filters=64, kernel_size=3,padding="causal", dilation_rate=8, activation='relu')(x)
      #x = tf.keras.layers.BatchNormalization()(x)
      #x = tf.keras.layers.Dropout(rate=0.1)(x)
      #x = tf.keras.layers.Add()([residual_3,x])

      #xout = tf.keras.layers.Conv1D(filters=4, kernel_size=1, activation='relu', name="CONV1_FINAL")(x)


# Load the UK-DALE dataset
appliance_name = input('Enter a value : ')


data =createdatacsv()

#data['time']=pd.to_datetime(data['time'], unit='s')
appliance=appliance_name
# Preprocess data for each appliance
#preprocessed_data = []
# Esempio di utilizzo
#raw_data = ...  # Dati grezzi di input
#time_column = ...  # Colonna contenente i valori di tempo
#target_column = ...  # Colonna contenente i valori target (y)
threshold = 10  # Soglia per bilanciare gli stati on/off
# Rimozione dei dati mancanti
#processed_data = remove_missing_data(data)
data=remove_missing_data(data)
#processed_data=gap_over(data)

processed_data=interpolation(data)
################# NON CONSIDERO LO SPILIT IN SOTTO SEQUENZE 
    # Divisione delle sequenze in sotto-sequenze
#subsequences = split_subsequences(processed_data, data['time'], 20)


################# AGGIUNGO 0 per buchi superiori a 2 minuti     ###################
#test = []
# Supponiamo che df sia il tuo dataframe
#indici_attributo_maggiore_di_10 = data.loc[data['data_df_appliance'] > 10].index.tolist()
#print("vettore 10 watt" ,len(indici_attributo_maggiore_di_10))
    


################# NON CONSIDERO IL BILANCIAMENTO 
    # Bilanciamento delle sequenze
threshold=10
#x_balance , y_balance = balance_samples(data['data_df_aggregate'], data['data_df_appliance'], threshold)




# Resetta l'indice del dataframe
#x_balance = x_balance.reset_index(drop=True)
#y_balance = y_balance.reset_index(drop=True)


    # Concatenazione dei dati di input e normalizzazione
#input_data = [sample['data_df2'] for subseq in balanced_subsequences for sample in subseq]

#### NORMALIZZAZIONE
#normalized_input = y_balance
#normalized_aggregate = x_balance
normalized_input = data['data_df_appliance']
normalized_aggregate = data['data_df_aggregate']
    # Stampa dei dati pre-processati
#print(normalized_input)
    #preprocessed_data = preprocess_data(data, appliance, parametri_apparati[appliance_name]['step_size'])
    #df=pd.array(preprocessed_data)
    #for array in preprocessed_data:
    #    tuttiarray=[]
    #    tuttiarray.append(array)
    #preprocessed_data = np.concatenate((tuttiarray), axis=0)
    #preprocessed_data=np.array(preprocessed_data)

    # Creazione del DataFrame con i dati preelaborati
    #data = preprocessed_data

    # Rimuovi le righe duplicate dal DataFrame
    #data = pd.DataFrame(data).drop_duplicates()
    #print(preprocessed_data)
    # Inizializzazione del DataFrame
    
    #time=np.array(list(map(numeric, data['time'])))
#aggregate=np.array(list(map(float, normalized_aggregate)))
   # len(normalized_input)
data = {
    'time': data['index'][:len(normalized_input)],
    'aggregate': normalized_aggregate[:len(normalized_input)],
    'data_appliance':normalized_input
    
}
dataframe = pd.DataFrame(data)

#dataframe=pd.DataFrame(preprocessed_data)
print(dataframe);
h=parametri_apparati[appliance_name]['train_build']
dataframe.to_csv(os.path.join("C:\\", "Users","silve","OneDrive","Desktop",str(appliance)+str("_NormalizeAggregate_GroundTruth")+ str("_train") +".csv"))
print("file salvato correttamente")
#savefile(dataframe,parametri_apparati[appliance_name]['train_build']+'house_') 
   
     
#else:
#    # Normalize the data
#    scaler = MinMaxScaler()
#    normalized_data = scaler.fit_transform(data)

#    # Standardize the data
#    standardizer = StandardScaler()
#    standardized_data = standardizer.fit_transform(normalized_data)

#    # Print the shape of the preprocessed data
#    print(standardized_data.shape)
#      #df=pd.array(preprocessed_data)
#    dataframe=pd.DataFrame(standardized_data)
#    print(dataframe);
#    h=parametri_apparati[appliance_name]['train_build']
#    savefile(dataframe,'aggregate_'+'house_' + str(h))  










#def preprocess_data(df, appliance, step_size):
#    # Remove missing data
#    df = df.dropna()
#    #range(len(df['time']))
#    # Split sequence into subsequences if less than 20 seconds
#    #range(len(data['time'])):
#    subsequences = []
#    subseq = []
#    for i in  range(100000):
#        subseq.append(pd.Series(df['data_appliance'].iloc[i]))
#        if((i < len(df['time']) - 1) and ((df['time'].iloc[i+1] - df['time'].iloc[i])).seconds >=20) :
#        #if (i < len(pd.to_datetime(df['time'])) - 1) and (pd.to_datetime(df['time'].iloc[i + 1]) - pd.to_datime(df['time'].iloc[i])).seconds >= 20:
#            subsequences.append(np.concatenate(subseq))    
#            np.concatenate(subsequences)
#            subseq = []
#    subsequences.append(np.concatenate(subseq))    
#    subsequences=np.concatenate(subsequences)


 

#    # Apply sliding window with step size
#    #window_size = 600
#    #windows = []
#    #for subseq in subsequences:
#    #    start_idx = 0
#    #    end_idx = window_size
#    #    while end_idx <= len(subseq):
#    #        windows.append(subseq.iloc[start_idx:end_idx])
#    #        start_idx += step_size
#    #        end_idx += step_size

#    # Randomly keep samples where y(t) >= 10 watts to balance on and off states
#    filtered_windows = []
#    for window in subsequences:      
#            if(window >10).any():
#                filtered_windows.append(window)

#    # Normalization of input samples (x)
#    normalized_windows = []
#    for window in filtered_windows:
#        max_value = window.max()
#        normalized_window = window.copy()
#        normalized_window= normalized_window / max_value
#        normalized_windows.append(normalized_window)
        
#    # Context information preservation using k > 1 delay monitoring (k = 200  UK-DALE)
#    #k = 200
        
#    #input_data = []
#    #target_data = []
#    #for window in normalized_windows:
#    #    if len(window) >= k + window_size:
#    #        input_data.append(window.iloc[-(k + window_size):-window_size].values)
#    #        target_data.append(window.iloc[-window_size:].values)
   
#    return normalized_windows