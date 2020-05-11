from bb_module_default import *

def main():
    "Local test batch to verify functionality"
    test = blackbox_function_airbnb(
        {
         'bedrooms':2,
         'bathrooms':2,
         'size': 40,
         'accommodates': 4,
         'distance': 3,
         'cleaning_fee': 20
        })

    print(test)

if __name__ == "__main__":
    main()
