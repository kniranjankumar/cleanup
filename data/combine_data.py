import pickle as pkl
data = []
for i in range(4):
    for j in range(10):
        with open('./collected_data/subject_'+str(i+1)+'/eps_'+str(j+1)+'.pkl','rb') as f:
            data.extend(pkl.load(f))
with open('./combined.pkl','wb') as f:
    pkl.dump(data,f)
