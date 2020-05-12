# Predict Airbnb Listing Prices: A Kinetica blackbox example using regression with Xgboost

More information can be found at: [Kinetica Documentation](http://www.kinetica.com/docs/7.0/index.html)

The Kinetica Blackbox SDK assists users in creating blackbox models to wrap existing code/functionality and make it deployable within the Kinetica system. The Active Analytics Workbench (AAW) currently can only import blackbox models that have been containerized and implement the BlackBox SDK. Users provide the module scripts, modify some SDK files, and the SDK will build a Docker Container from the files and publish it to a given Docker Registry (private or public).

In this example, we take historical, public Airbnb data and train a regressor to predict future listing prices that come to market using Kinetica's Active Analytics Workbench.

## Prerequisites

This tutorial assumes you have a Docker account and a Kinetica instance with the Active Analytics Workbench package installed. If you don't you can **download a free trial here** - https://www.kinetica.com/trial/
* [Docker](https://www.docker.com/get-started)
* Docker [Hub](https://hub.docker.com/) / Docker Registry

## Download and Configuration

First create a new project. For this example I used Pycharm and am using a Conda enviroment. Download this repository, install the Kinetica Python API, and install any initial libraries you will use. 

1. Clone the project and install the Kinetica Python API:

        git clone https://github.com/nickalonso/kml-bbox-tutorial.git
        pip install gpudb --upgrade
        

## Setup

This repository contains the files needed to build and publish a blackbox model to a Docker container compatible with AAW. The important files and their function, as well as the necessary data sets to train and deploy a machine learning model with Kinetica are included:

**WARNING:** It's highly recommended the ``sdk/*`` and ``bb_runner.sh`` files are not modified!

| Filename                    | Description                                                                                                                                                           |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sdk/bb_runner.py`          | Python script called from the Docker container entrypoint script. Contains the code necessary for the module(s) to interface with the `kinetica_black_box.py` script. |
| `sdk/kinetica_black_box.py` | Python script called from `bb_runner.py`. Contains the code necessary for the blackbox module(s) to interface with the database.                                      |
| `Dockerfile`                | File containing all the instructions for Docker to build the model image properly.                                                                                    |
| `bb_module_default.py`      | Python script containing model code. The default code is a template for you to reuse and/or replace.                                                                  |
| `bb_runner.sh`              | Entrypoint for the Docker container; this script will be run initially when AAW pulls the container for execution.                                                    |
| `release.sh`                | Utility script for building and publishing the model to a Docker Hub or Docker Registry.                                                                              |
| `model_training.py`                |  A blank script to conduct model selection and training.                                                                               |
| `requirements.txt`          | Text file that stores the required python libraries for the model. Default libraries (`gpudb`, `zmq`, `requests`) must be left intact.                                |
| `data/airbnb_historical_listings.csv` |  Training data |                                                                                            
| `data/airbnb_new_listings.csv`     |  A small table without the target variable (price) that can be passed as a batch inference.  
                                                   


To setup the repository for publishing your model:

1. The file `bb_module_default.py` has been pre-built for you. This file can contain as many methods as desired or call as many other modules as desired, but the default method **must** take a dictionary in (`inMap`) and return a dictionary (`outMap`). This is the file that will recieve input from AAW, call our pre-trained model and make the inference, then return the results.

2. Optionally, update the name of `bb_module_default.py`. If the module name is updated, it will need to be referenced appropriately when deploying the model via the AAW UI or the AAW REST API. See the *Usage* section for more information.

3. Update `model_training.py` with your training and development. In this example I saved my model using **pickle**  

4. Open the `Dockerfile` i and include any required installations that are not easily installable with `pip`:

        RUN apt-get install -y git wget

5. Add all module files:

        ADD <module file.py> ./

   **IMPORTANT:** By default, the `Dockerfile` includes a reference to `bb_module_default.py`. This reference **must** be updated if the file name was changed earlier.

1. Open `requirements.txt` and include any additional required python libraries::

        gpudb==7.0.3.0
        zmq
        requests
        xgboost
        pandas
        scikit-learn

   **IMPORTANT:** The default `gpudb`, `zmq`, and `requests` packages inside `requirements.txt` **must** be left in the file.

1. Open `release.sh` in a text editor and update the repository, image,  and tag for both the `build` and `push` statements:

        docker build -f Dockerfile -t <repo-name>/<image-name>:<tag-name> .
        docker push <repo-name>/<image-name>:<tag-name>

   **TIP:** The Docker repository will be created if it doesn't exist.

## Usage

### Publishing the Model

1. Login into your Docker Hub or Docker Registry:

        # Docker Hub
        docker login

        # Docker Registry
        docker login <hostname>:<port>

2. Run the `release.sh` script to build a Docker image of the model and publish it to the provided Docker Hub or Docker Registry:

        ./release.sh

### Importing the Model

After publishing the model, it can be imported into AAW using two methods:

* REST API (via `cURL`)
* AAW User Interface (UI)

#### REST API

If using the REST API, a model is defined using JSON. The `cURL` command line tool can be used to send a JSON string or file to AAW. To import a blackbox model into AAW using ``cURL`` and the REST API: 

1. Define the model. *Kinetica* recommends placing the model definition inside a local JSON file.
1. Post the JSON to the `/model/blackbox/instance/create` endpoint of the AAW REST API:

        # Using a JSON file
        curl -X POST -H "Content-Type: application/json" -d @<model_file>.json http://<kinetica-host>:9187/kml/model/blackbox/instance/create

        # Using a JSON string
        curl -X POST -H "Content-Type: application/json" -d '{"model_inst_name": "<model_name>", ... }' http://<kinetica-host>:9187/kml/model/blackbox/instance/create

To aid in creating the necessary JSON, use the following endpoint and schema:

**Endpoint name**: ``/model/blackbox/instance/create``

**Input parameters**:

| Name | Type | Description |
|---------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `model_inst_name` | string | Name of the model. |
| `model_inst_desc` | string | Optional description of the model. |
| `problem_type` | string | Problem type for the model. Always `BLACKBOX`. |
| `model_type` | string | Type for the model. Always `BLACKBOX`. |
| `input_record_type` | array of map(s) of strings to strings | An array containing a map for each input column. Requires two keys. Valid key name, type, and descriptions found below. |
| `model_config` | map of strings to various | A map containing model configuration information. Valid key name, type, and descriptions found below. |

Array `input_record_type` of map keys:

**IMPORTANT:** There will need to be as many maps (containing both name and type) as there are columns in the `inMap` variable inside the default blackbox module.

| Name | Type | Description |
|------------|--------|----------------------------|
| `col_name` | string | Name for the input column. |
| `col_type` | string | Type for the input column. |

Map `model_config` keys:

**IMPORTANT:** There will need to be as many maps (containing both name and type) in `output_record_type` as there are columns in the `outMap` variable inside the default blackbox module.

| Name | Type | Description |
|----------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `db_user` | string | Username for database authentication. |
| `db_pass` | string | Password for database authentication. |
| `blackbox_module` | string | Module name for the blackbox model. |
| `blackbox_function` | string | Function name inside the blackbox module. |
| `container` | string | Docker URI for the container, e.g., `<repo_name>/<image_name>:<tag_name>` |
| `output_record_type` | string | An array containing a map for each output column. Similar to `input_record_type`, requires two keys: `col_name` -- a string value representing the name of the output column; `col_type` -- a string value representing the type of the output column. |

**Example JSON**:

The final JSON string should look similar to this:

    {
      "model_inst_name": "Taxi Fare Predictor",
      "model_inst_desc": "Blackbox model for on-demand deployments",
      "problem_type": "BLACKBOX",
      "model_type": "BLACKBOX",
      "input_record_type": [
        {
          "col_name": "pickup_longitude",
          "col_type": "float"
        },
        {
          "col_name": "pickup_latitude",
          "col_type": "float"
        },
        {
          "col_name": "dropoff_longitude",
          "col_type": "float"
        },
        {
          "col_name": "dropoff_latitude",
          "col_type": "float"
        }
      ],
      "model_config": {
        "db_user": "",
        "db_pass": "",
        "blackbox_module": "bb_module_default",
        "blackbox_function": "predict_taxi_fare",
        "container": "kinetica/kinetica-blackbox-quickstart:7.0.1",
        "output_record_type": [
          {
            "col_name": "fare_amount",
            "col_type": "double"
          }
        ]
      }
    }

#### AAW User Interface (UI)

The AAW UI offers a simpler WYSIWYG-style approach to importing a blackbox model. To import a blackbox model into the UI:

1. Navigate to the AAW UI (`http://<aaw-host>:8070`)
1. Click `Models + Analytics`.
1. Click `+ Add Model` then `Import Blackbox`.
1. Provide a `Model Name` and optional `Model Description`.
1. Input the Docker URI for the container, e.g., `<repo_name>/<image_name>:<tag_name>`
1. Input the `Module Name` and `Module Function`.
1. For `Input Columns`:

   1. Click `Add Input Column` to create input columns.
   1. Provide a `Column` name and `Type`.

1. For `Output Columns`:

   1. Click `Add Output Column` to create output columns.
   1. Provide a `Column` name and `Type`.

1. Click:`Create`.


