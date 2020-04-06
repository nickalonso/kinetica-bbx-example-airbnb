#! /bin/bash
docker build -f Dockerfile -t nickalonso/ctnr-bbox-airbnb-new:v1 .

if [ "$?" -eq "0" ]
then
	echo "Docker Build Successful, Publishing container publicly"
	docker push nickalonso/ctnr-bbox-airbnb-new:v1
else
 	echo "Docker Build Failed, no release executed"
fi
