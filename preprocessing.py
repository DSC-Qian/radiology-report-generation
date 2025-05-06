import pandas as pd

record = pd.read_csv('cxr-record-list.csv')
study = pd.read_csv('cxr-study-list.csv')

data1 = pd.read_csv('/content/mimic-cxr-2.0.0-metadata.csv')

df_metadata = data1.copy()

VIEW_MAP = {
    'AP': 'frontal',
    'PA': 'frontal',
    'LATERAL': 'lateral',
    'LL': 'lateral',
    'LPO': 'other',
    'RAO': 'other',
    'RPO': 'other',
    'LAO': 'other',
    'AP AXIAL': 'other',
    'XTABLE LATERAL': 'other',
    'AP LLD': 'other',
    'PA LLD': 'other',
    'L5 S1': 'other',
    'SWIMMERS': 'other',
    'AP RLD': 'other',
    'PA RLD': 'other',
}

df_metadata['view'] = df_metadata['ViewPosition'].map(VIEW_MAP)

good_view = ['frontal', 'lateral']
data2 = df_metadata[df_metadata['view'].isin(good_view)]

data3 = data2[data2['view'] == 'frontal'].drop_duplicates(subset='study_id', keep='first')

data4 = data3[['subject_id','study_id','dicom_id']].merge(record, how = 'inner', on = ['subject_id','study_id','dicom_id'])
data4['image_path'] = data4['path'].str.replace('files', 'images')
data4['image_path'] = data4['image_path'].str.replace(r'\.dcm$', '.jpg', regex=True)
data4 = data4.drop(columns = ['path'])

data5 = data4.merge(study, how = 'inner', on = ['subject_id','study_id'])
data5['report_path'] = data5['path'].str.replace('files', 'reports')
data5 = data5.drop(columns = ['path'])

data5.to_csv('mimic-cxr-list-filtered.csv', index = False)