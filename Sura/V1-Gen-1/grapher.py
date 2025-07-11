# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the Excel file
# file_path = '20-21-22-24 May Data Analysis.xlsx'
# xls = pd.ExcelFile(file_path)
# df_5_june = pd.read_excel(file_path, sheet_name='5 june')


# df_5_june_100_2=df_5_june[3:59].loc[:,('Unnamed: 1','Unnamed: 3')]
# print(df_5_june_100_2)
# fig,ax=plt.subplots()
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Viscosity")
# plt.ylabel("Shear Rate")
# plt.title("Plot for 5 June")
# ax.plot(df_5_june_100_2['Unnamed: 1'],df_5_june_100_2['Unnamed: 3'])
# ax.grid()
# fig.savefig('5june_100_20.png')
# # print(df_5_june[3:59].loc[:])

# import matplotlib.pyplot as plt
# import numpy as np

# xpoints = np.array([1, 8])
# ypoints = np.array([3, 10])

# plt.plot(xpoints, ypoints)
# plt.show()



