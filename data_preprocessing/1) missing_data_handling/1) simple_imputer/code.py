import numpy as np
data = np.array([[2.3, np.nan], [np.nan, 3], [4, 5]])
print("Original data:\n", data)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data)
print("transformed_data:\n",imputer.transform(data))

# execution: run this file on the terminal to get output. which would look something like below:
'''
Original data:
 [[2.3 nan]
 [nan 3. ]
 [4.  5. ]]
transformed_data:
 [[2.3  4.  ]
 [3.15 3.  ]
 [4.   5.  ]]
'''

