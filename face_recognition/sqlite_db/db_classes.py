from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String

Base = declarative_base()

class FacePattern(Base):
    __tablename__ = 'face_pattern'
    id = Column(Integer, primary_key=True)
    file_name = Column(String)
    file_hash = Column(Integer)
    pattern_identity = Column(String)
    encodings = Column(String)