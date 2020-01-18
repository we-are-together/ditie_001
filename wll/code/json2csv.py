import json
import csv
import glob
import os
import re
class Json_Csv:
    def exchange(self):
        all_files = glob.glob(os.path.join("*.json"))
        for file in all_files:
            with open('{}'.format(file), 'r', encoding='utf-8', errors='ignore') as f:
               a = json.load(f)  # 此时a是一个字典对象
               list1 = a['Mapping']
            result=re.match('(.+)\.json',file)
            f=result.group(1)
            csvFile = open("{}.csv".format(f), 'w', newline='', encoding='utf-8')
            writer = csv.writer(csvFile)
            writer.writerow(('ElementName','FamilyName','CategoryName','ItemName','MinX','MinY','MinZ','MaxX','MaxY','MaxZ','Length','Depth','Height','Volume','Surface'))
            for dict in list1:
               ElementName = dict['ElementName']
               FamilyName = dict['FamilyName']
               CategoryName = dict['CategoryName']
               ItemName = dict['ItemName']
               MinX = dict['Min']['X']
               MinY = dict['Min']['Y']
               MinZ = dict['Min']['Z']
               MaxX = dict['Max']['X']
               MaxY = dict['Max']['Y']
               MaxZ = dict['Max']['Z']
               Length = MaxX - MinX
               Depth = MaxY - MinY
               Height = MaxZ - MinZ
               Volume = Length * Depth * Height
               Surface = (Length * Depth) * 2 + (Length * Height) * 2 + (Depth * Height) * 2
               writer.writerow((ElementName, FamilyName, CategoryName, ItemName, MinX, MinY, MinZ, MaxX, MaxY, MaxZ, Length, Depth, Height, Volume, Surface))
JC=Json_Csv()
JC.exchange()



