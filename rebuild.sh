#!/bin/bash

docker stop strm2
docker build -t anomacti/strm1 .
docker run -d --rm -p 8502:8501 -v /home/anomacti/Dev/University/software/data-analysis-web:/app --name strm2 anomacti/strm1
