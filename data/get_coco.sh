curl "http://images.cocodataset.org/zips/train2014.zip" --output train2014.zip
mv train2014.zip coco/
curl "http://images.cocodataset.org/zips/val2014.zip" --output val2014.zip
mv val2014.zip coco/
curl "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" --output ann_trainval.zip
mv ann_trainval.zip coco/
