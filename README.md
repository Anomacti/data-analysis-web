# data-analysis-web

A web application for analysing tabular data and performing machine learning algorithms.

## Project members

- Ανανίας Χορόζογλου - Π2020019

## Instructions for running the project

### Docker build

To build the docker image run the following command:

`docker build -t uni/strm1 .`

### Docker run

To run the created image use the following command:

`docker run -d --rm -p 8502:8501 --name strm2 uni/strm1`

### Opening application

After running those two command the application will start running on port `8502` on localhost.

To open the application go to [localhost](http://localhost:8502)

### Stoping application

To stop the application run:

`docker stop strm2`
