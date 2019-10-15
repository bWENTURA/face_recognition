# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import json
import zlib
import PIL.Image
import numpy as np
from sqlalchemy import create_engine
import re
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlite_db.db_classes import Base, FacePattern

def scan_known_people(known_people_folder):
    basename_pattern = re.compile(r"[a-zA-Z]+\_[a-zA-Z]+")
    basename_pattern_number = re.compile((r"[a-zA-Z]+\_[a-zA-Z]+\_[0-9]+"))
    name_surname_pattern = re.compile((r"[a-zA-Z]+"))

    db_engine = create_engine("sqlite:///face_recognition/sqlite_db/face_recognition.db")
    Base.metadata.create_all(db_engine)
    session_factory = sessionmaker(bind=db_engine)
    session = scoped_session(session_factory)

    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        
        # Check for files with appropriate name
        if basename_pattern.fullmatch(basename) or basename_pattern_number.fullmatch(basename):
            # Get name and surname of the person from image
            pattern_identity = ' '.join([name.capitalize() for name in name_surname_pattern.findall(basename)])

            # Select pattern from database
            face_pattern = session.query(FacePattern).filter_by(file_name = basename).first()
            if face_pattern:
                if face_pattern.file_hash == zlib.crc32(open(file, "rb").read()):
                    click.echo("Pattern for {} found in database.".format(file))
                    known_names.append(face_pattern.pattern_identity)
                    known_face_encodings.append(np.array(json.loads(face_pattern.encodings)))
                else:
                    click.echo("Pattern for {} found in database, but CRC32 is not the same." \
                         "Probably someone have overwritten the file.".format(file))
            else:
                click.echo("Pattern for {} not found in database. Calculating pattern..".format(file))

                encodings = face_recognition.face_encodings(img)

                if len(encodings) > 1:
                    click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

                if len(encodings) == 0:
                    click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
                else:
                    known_names.append(pattern_identity)
                    known_face_encodings.append(encodings[0])
                    session.add(
                        FacePattern(
                            file_name = basename,
                            file_hash = zlib.crc32(open(file, "rb").read()),
                            pattern_identity = pattern_identity,
                            encodings = json.dumps(list(encodings[0]))
                        )
                    )
                    session.commit()

    return known_names, known_face_encodings


def print_result(filename, name, distance, show_distance=False):
    if show_distance:
        print("{},{},{}".format(filename, name, distance))
    else:
        print("{},{}".format(filename, name))


def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:
            [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
        else:
            print_result(image_to_check, "unknown_person", None, show_distance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, "no_persons_found", None, show_distance)


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.argument('known_people_folder')
@click.argument('image_to_check')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
def main(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)


if __name__ == "__main__":
    main()
