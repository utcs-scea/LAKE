mvNCCompile -s 12 ./data/inception_v3_2016_08_28_frozen.pb -in=input -on=InceptionV3/Predictions/Reshape_1
mv graph data/inception_v3_movidius.graph
