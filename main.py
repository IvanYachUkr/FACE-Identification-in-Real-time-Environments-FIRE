# main.py
# modules/
#     __init__.py
#     encryption.py
#     database.py
#     hnsw_manager.py
#     detector.py
#     encoder.py
#     face_recognition.py
#     tracker.py
#     utils.py
#
# The main.py file handles argument parsing, instantiation of the FaceRecognition class, and
# calling the corresponding processing functions. The modules folder contains separate files
# for different functionalities.
#
# --------------------------------------------
# File: main.py
# --------------------------------------------
import argparse
import logging
from modules.face_recognition import FaceRecognition
from modules.utils import set_single_core_affinity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System with Encryption and Custom Filenames")
    parser.add_argument('--mode', type=str, default='image',
                        choices=['image', 'video', 'webcam'],
                        help='Mode of operation: image, video, or webcam')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or video file')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the annotated output')
    parser.add_argument('--label', type=str, default=None,
                        help='Label/name for adding a new face (for image mode) or renaming recognized faces (webcam mode)')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging')
    parser.add_argument('--show', action='store_true',
                        help='Enable display of processed frames')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for encrypting/decrypting files')
    parser.add_argument('--detector', type=str, default="mediapipe",
                        choices=["mediapipe", "yunet", "retinaface"],
                        help='Detector type: mediapipe, yunet, retinaface')
    parser.add_argument('--encoder', type=str, default=None,
                        choices=["128", "512"],
                        help='Encoder type: "128" for Facenet 128 and "512" for Facenet 512')
    parser.add_argument('--detection_interval', type=int, default=1,
                        help='Number of frames to skip for face detection, use 1 for images')
    parser.add_argument('--core', type=int, default=0,
                        help='0 use all cores, 1 use 1 core')

    annotate_group = parser.add_mutually_exclusive_group()
    annotate_group.add_argument('--annotate', dest='annotate', action='store_true',
                                help='Enable drawing bounding boxes and labels')
    annotate_group.add_argument('--no-annotate', dest='annotate', action='store_false',
                                help='Disable drawing bounding boxes and labels')
    parser.set_defaults(annotate=True)

    parser.add_argument('--hnsw_index_path', type=str, default=None,
                        help='Custom path for the HNSWlib index file')
    parser.add_argument('--hnsw_labels_path', type=str, default=None,
                        help='Custom path for the HNSWlib labels file')
    parser.add_argument('--hnsw_db_ids_path', type=str, default=None,
                        help='Custom path for the HNSWlib DB IDs file')
    parser.add_argument('--sqlite_db_path', type=str, default=None,
                        help='Custom path for the SQLite database file (unencrypted)')
    parser.add_argument('--sqlite_db_encrypted_path', type=str, default=None,
                        help='Custom path for the encrypted SQLite database file')
    parser.add_argument('--interested_label', type=str, default=None,
                        help='If set, only faces with this label will be recognized/maintained')

    args = parser.parse_args()

    face_recog = FaceRecognition(
        detector_type=args.detector,
        align=False,
        encoder_model_type=args.encoder,
        encoder_mode='cpu_optimized',
        similarity_threshold=0.7,
        enable_logging=args.log,
        show=args.show,
        unknown_trigger_count=1,
        detection_interval=1 if args.mode == "image" else args.detection_interval,
        encryption_password=args.password,
        hnsw_index_path=args.hnsw_index_path,
        hnsw_labels_path=args.hnsw_labels_path,
        hnsw_db_ids_path=args.hnsw_db_ids_path,
        sqlite_db_path=args.sqlite_db_path,
        sqlite_db_encrypted_path=args.sqlite_db_encrypted_path,
        interested_label=args.interested_label
    )

    if args.core:
        print(f"Use only 1 cpu core: {bool(args.core)}")
        set_single_core_affinity()

    if args.mode == 'image':
        if args.input is None:
            logging.error("Please provide the path to the input image using --input")
        else:
            face_recog.process_image(
                image_path=args.input,
                annotate=args.annotate,
                save_path=args.save,
                label=args.label
            )

    elif args.mode == 'video':
        if args.input is None:
            logging.error("Please provide the path to the input video using --input")
        else:
            face_recog.process_video(
                video_path=args.input,
                annotate=args.annotate,
                save_path=args.save
            )

    elif args.mode == 'webcam':
        face_recog.process_webcam(
            annotate=args.annotate,
            save_path=args.save,
            name=args.label
        )

    if args.label:
        face_recog.shrink_db_ids()

    face_recog.close()

