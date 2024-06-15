import numpy as np
import Databases

if __name__ == '__main__':
    # db_names = ['BUPA', 'DERMATOLOGY', 'DIABETES', 'HEART', 'WAVEFORM', 'WDBC', 'GAUSS50', 'GAUSS50x']
    db_names = ['BUPA', 'DERMATOLOGY', 'DIABETES', 'HEART', 'WAVEFORM', 'WDBC']
    for db_name in db_names:
        X, Y, label = Databases.select_database(db_name)
        print(f"Database: {label}")
        print(f"X shape: {X.shape}")
        print(f"Y classes: {len(np.unique(Y))}")
        print()

# Database: BUPA
# X shape: (345, 5)
# Y classes: 2
#
# Database: DERMATOLOGY
# X shape: (358, 33)
# Y classes: 6
#
# Database: DIABETES
# X shape: (768, 8)
# Y classes: 2
#
# Database: HEART
# X shape: (297, 13)
# Y classes: 2
#
# Database: WAVEFORM
# X shape: (1000, 21)
# Y classes: 3
#
# Database: WDBC
# X shape: (568, 30)
# Y classes: 2

