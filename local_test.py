from bb_module_default import *

"Demonstrates a typical data science training environment with 0 auditing capabilities"
def main():
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