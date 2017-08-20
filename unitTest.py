from main import Images2Dataset as im2da
import os 

dataset = im2da(dbFolder = os.getcwd()+"/tests/db/")

df = dataset.uris2Dataframe()

print(df)