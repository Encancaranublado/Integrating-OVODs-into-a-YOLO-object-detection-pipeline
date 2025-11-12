# Integrating-OVODs-into-a-YOLO-object-detection-pipeline

This repository contains experimental work relating to the intersection of computer vision/remote sensing, and natural language processing.

# Abstract 

  Advances in both computer vision and natural language processing force us to tackle a variety of concerns. Most notably, how to we use methods in NLP to improve Computer Vision pipelines and vice-versa? 
This experimental project was built with this question in mind, with special attention to simplicity and demonstration of the effects and roles of the moving parts of the system. 

  We use an idea of the teacher-student setup in which we train one base YOLOv5 model (pretained on the COCO dataset) and trained it on the xView 2018 dataset, which is a well-known remote sensing dataset. Post-training, we extract its pseudo-labels by running the YOLOv5 model on the training data. Once these pseudo-labels are extracted, we investigate for gaps in them; due to the fact that even great deep learning models may miss objects that are barely seen (xView has vanishingly small categories in terms of some of their object instances). Once the gaps were assessed, we performed a backfill from the gold-labeled dataset to augment the pseudo-label training set with categories that are not found in the pseudo-labels. 

# Open Vocabulary Object Detectors

  After the backfill, we perform a series of experiments. The original pseudo-labels of the training set are run through individual pipelines where we use two different open-vocabulary object detectors to audit and filter the data, with the hopes that the student YOLOv5 models will find it easier to learn from that data. We used CLIP to attempt to filter false positives by using its similarity vector space, comparing the category names to what it can observe in the labeled object. It is important to mention that the object (as seen from an overhead perspective) is passed into CLIP cropped from the image. This was a calculated risk, since crops of objects do not always preserve the context needed to define the object, not to mention the reduction in resolution quality which may make it harder for the OVODs to detect the objects at all. 

  We used OWLv2 for a different purpose; to modify and snap bounding boxes more snugly around the object. We did not filter objects out, but we added some moderately regulated gates to mitigate, in case OWLv2 makes mistakes, the risk of erroneous bounding boxes and therefore worsening mAP metrics. We added similar cosine similarity gates to CLIP, since we wanted to reduce the risk of true positives being dropped. The final pipeline combines these two approaches by passing the pseudo-labels through both OVODs. 

# Brief report of results

  In the end, we found that using OWLv2 positively impacts every single performance metric, CLIP dropped a significant number of TNs thereby worsening performance, and the combination of both results in an evening-out. We trained 4 YOLOv5 models: one on the original pseudo-labels, another in the OWLv2 modified labels, another in the CLIP filtered labels, and finally pseudo-labels that were modified by both aforementioned OVODs. The YOLOv5 model trained and tested on the OWLv2 modified pseudo-labels performed the best out of all the student models, while the CLIP-only student performed worse. 







