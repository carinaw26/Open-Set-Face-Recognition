'''
Prerequisities:
  numpy
  opencv-utils
  opencv-python
  opencv-python-headless
'''
import sys
import os
import os.path
import getopt
import numpy as np
import cv2 as cv2


class ExtracFaces(object):
  '''
  Extract faces from input images by using  Haar Cascade through OpenCV
  https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python
  '''

  def __init__(self):
    '''
    The default constructor
    '''
    self.fileName        = ""
    self.fileDirName     = ""
    self.inputDirectory  = ""
    self.recursive        = False
    self.outputDirectory = ""
    self.outputSuffix    = 'face'
    self.faceIndex       = 1
    self.extensions       = []
    self.scaleFactor      = 1.3
    self.minNeighbors     = 5
    self.minHeight        = 30
    self.minWidth         = 30
    self.cascade          = "haarcascade_frontalface_default.xml"
    self.faceCascade      = None
    self.faceHeight       = 224
    self.faceWidth        = 224 
    self.outputFace      = False

  def usage (self):
    print("Usage ExtractFaces.py [option(s)]\n")
    print("  -?, --help              print help")
    print("  -c, --cascade           The default is haarcascade_frontalface_default.xml")
    print("  -F, --output-face       output faces instead of orginal images with detected faces")
    print("  -f, --input-file        input image file. For example, /tmp/s123.bmp")
    print("  -e, --extensions        list of extension seperated by ','. The default will be all image formats supported by imghdr.what()")
    print("  -H, --face-height       face output hight. The default is 224")
    print("  -h, --min-height        minimum extract area height. The default is 40")
    print("  -i, --input-directory   input directory. For example. /tmp/input")
    print("  -n, --min-neighbors     minimum extract area neighbors. The default is 3")
    print("  -o, --output-directory  output directory")
    print("  -r, --recursive         recursively find leave directory to find input images")
    print("  -s, --suffix            output name suffix as format <original name>-<suffix>-<index>.<original-extension>")
    print("                          The default is 'face'")
    print("  -W, --face-width        face output width. The default is 224")
    print("  -w, --min-width         minimum extract area width. The default is 40")
    print("  -x, --scale-factor      the default is 1.3")
    print("\nFor example:  python ./extract-faces.py -e bmp -i C:\Work\LexisNexis\HPCC\2021\intern\gnn-cloud\images\original -r -o ../../images/faces -F")
   
    exit(1)

  def process_args(self):
    try:
      opts, args = getopt.getopt(sys.argv[1:],":c:e:Ff:H:h:i:n:o:rs:W:w:x:",
        ["help", "cascade","input-file", "extensions","output-face", "face-height","min-height", "input-directory", "min-neighbors", 
         "output-directory","recursive", "suffix","face-width","min-width","scale-factor"])
                 
    except getopt.GetoptError as err:
      print(str(err))
      self.usage()

    for arg, value in opts:
      if arg in ("-?", "--help"):
        self.usage()
      elif arg in ("-c", "cascade"):
        self.cascade = value
      elif arg in ("-e", "extensions"):
        self.extensions = list(value.split(","))
      elif arg in ("-F", "ouput-face"):
        self.outputFace = True
      elif arg in ("-f", "input-file"):
        self.fileName = value
      elif arg in ("-H", "face-height"):
        self.faceHeight = value
      elif arg in ("-h", "min-height"):
        self.minHeight = value
      elif arg in ("-i", "input-directory"):
        self.inputDirectory = value
      elif arg in ("-n", "min-neighbors"):
        self.minNeighbors = value
      elif arg in ("-o", "output-directory"):
        self.outputDirectory = value
      elif arg in ("-r", "recursive"):
        self.recursive = True
      elif arg in ("-s", "suffix"):
        self.outputSuffix = value
      elif arg in ("-W", "face-width"):
        self.faceWidth = value
      elif arg in ("-w", "min-width"):
        self.minWidth = value
      elif arg in ("-x", "scale-factor"):
        self.scaleFactor = value
      else:
        print("\nUnknown option: " + arg)
        self.usage()

    if self.fileName:
      self.fileDirName  = os.path.dirname(self.fileName)
      self.fileName = os.path.basename(self.fileName)
      
      if not self.outputDirectory:
        self.outputDirectory = self.fileDirName

    self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + self.cascade)

  def validate_args(self):

    if (not os.path.isdir(self.inputDirectory)) and (not os.path.exists(os.path.join(self.fileDirName, self.fileName))):
      print("\nMust has either an existing file name or an existing input direcotry")
      exit(1)
    if not os.path.isdir(self.outputDirectory):
      print("\n\"" + self.outputDirectory + "\" is not a directory or does not exist.\n")
      exit(1)
  
  def extract_in_one_image(self, file, fileDir, outRelatedPath):
    imagePath = os.path.join(fileDir, file)
    #print(f"File: {file}, dir: {fileDir}")
    image = cv2.imread(imagePath)
    if (image is None):
      return
    
    fileName, fileExtension = os.path.splitext(file)
    extensionWithoutDot = fileExtension.strip('.')
    if (self.extensions and (extensionWithoutDot not in self.extensions)):
      return

    print(f"Process {imagePath}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = self.faceCascade.detectMultiScale(
      gray,
      scaleFactor=self.scaleFactor,
      minNeighbors=self.minNeighbors,
      minSize=(self.minHeight, self.minWidth)
    )
    print("[INFO] Found {0} Faces!".format(len(faces)))

    if (len(faces) <= 0):
      return

    # Output
    
    outputDir = os.path.join(self.outputDirectory, outRelatedPath)
    if (not os.path.isdir(outputDir)):
      os.mkdir(outputDir)

    index = 1
    for (x, y, w, h) in faces:
      if (self.outputFace):
        outputFile = os.path.join(outputDir, fileName + "-" + self.outputSuffix + "-" + str(index) +  "." + fileExtension)
        index += 1
        croppedImage = image[y:(y+h), x:(x+w)]
        status = cv2.imwrite(outputFile, cv2.resize(croppedImage, (self.faceWidth, self.faceHeight)))
        print("[INFO] Choffed image face " + outputFile + " written to filesystem: ", status)
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if ( not self.outputFace):
      outputFile = os.path.join(outputDir, fileName + "-" + self.outputSuffix + "." + fileExtension)
      status = cv2.imwrite(outputFile, image)
      #outputFile = os.path.join(self.outputDirectory, fileName + "-gray." + fileExtension)
      #status = cv2.imwrite(outputFile, gray)
      print("[INFO] Image face " + outputFile + " written to filesystem: ", status)

  def extract_all(self):
    if (os.path.exists(os.path.join(self.fileDirName, self.fileName))):
      self.extract_in_one_image(self.fileName, self.fileDirName, ""),
    if (os.path.isdir(self.inputDirectory)):
      for full_path, d_names, f_names in os.walk(self.inputDirectory):
        for f in f_names: 
          relatedPath  = full_path[len(self.inputDirectory):].strip(os.sep)
          #print(os.path.join(full_path, f) + ", " + relatedPath)
          self.extract_in_one_image(f, full_path, relatedPath)


if __name__ == '__main__':

  ef = ExtracFaces()
  ef.process_args()
  ef.validate_args()
  ef.extract_all()
  
  