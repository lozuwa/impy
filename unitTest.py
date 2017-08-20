# General purpose
import os 
# Main
from main import Images2Dataset as im2da
# Utils 
from utils import *
# Stats
from stats import *

DB_FOLDER = os.getcwd()+"/tests/db/"

def test_fillDictRows():
	dict_ = {'a': [1,2,3], 'b': [1,2]}
	dict_ = fillDictRows(dict_)
	print(dict_)

def test_uris2Dataframe():
	dataset = im2da(dbFolder = DB_FOLDER)
	df = dataset.uris2Dataframe(returnTo = True)
	print(df)

def test_tensorSizes():
	dataset = im2da(dbFolder = DB_FOLDER)
	df = dataset.uris2Dataframe(returnTo = True)
	eda = EDA(df)
	eda.tensorSizes()

def test_classesBalance():
	dataset = im2da(dbFolder = DB_FOLDER)
	df = dataset.uris2Dataframe(returnTo = True)
	eda = EDA(df)
	eda.classesBalance(visualize = True)

if __name__ == "__main__":
	# Which one would you like to test?
	test_classesBalance()