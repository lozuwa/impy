"""
package: Images2Dataset
class: MongoDb
Author: Rodrigo Loza
Description: Support class for images2dataset
"""
# General purpose 
import os
import sys
# Database
from pymongo import *

class MongoDb:
    """ 
    Connects to mongodb. Instantiates a new database. Supports write, read functions. 
    """
    def __init__(self, 
                name = "data", 
                create = False):
        """
        Constructor
        :param name: string that contains the database's name
        :param create: boolean that decides whether to create a new db or not
        """
        # Connect to the db
        self.client = MongoClient()
        #########TO FIX############
        self.db = client.data
        ###########################
        # If create is true, then start a new array
        if create:
            self.db.data.insert({"_id": 0, "images": []})

    def writeData(self, uri):
        """
        Write field
        : return: confirmation response  
        """
        assert type(uri) == str, "uri must be a string"
        self.db.data.update({"_id": 0}, {"images": uri}) 

    def readData(self, uri):
        """
        NOT IMPLEMENTED YET
        Find filed
        : return: uri
        """
        assert True == True, "function not implemented"
        assert type(uri) == str, "uri must be a string"
        return False

    def readAllData(self):
        """
        Reads all the database
        return: A list that contains all the uris
        """
        return [each for each in self.db.data.find()]

    def dropDb(self):
        """
        Careful!
        Eliminate database
        """
        self.db.data.drop()