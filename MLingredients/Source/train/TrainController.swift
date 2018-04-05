import UIKit
import PlanetSwift
import CoreML
import Vision

class TrainController: PlanetViewController, CameraCaptureHelperDelegate {
    
    let ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,"
    
    var captureHelper = CameraCaptureHelper(cameraPosition: .back)
    var model:VNCoreMLModel? = nil
    var overrideImage:CIImage? = nil
    let ciContext = CIContext(options: [:])
    var lastOriginalImage:CIImage? = nil
    
    var ocrModel:VNCoreMLModel? = nil
    
    var lastImageUsedForObservations:CIImage? = nil
    var observationsToProcess:[Any?] = []
    
    let trainingImagesPublisher:SwiftyZeroMQ.Socket? = Comm.shared.publisher(Comm.endpoints.pub_TrainingImages)
    
    
    func showCurrentObservation() {
        
        //print(self.observationsToProcess)
        
        if observationsToProcess.count == 0 {
            clearObservations()
            return
        }
        
        let perspectiveImagesCoords = self.observationsToProcess[0]
        if perspectiveImagesCoords == nil {
            goToNextObservation()
            return
        }
        
        DispatchQueue.main.async {
            let extractedImage = self.lastOriginalImage!.applyingFilter("CIPerspectiveCorrection", parameters: perspectiveImagesCoords as! [String : Any])
            let scaledImage = extractedImage.transformed(by: CGAffineTransform.init(scaleX: 28.0 / extractedImage.extent.size.width, y: 28.0 / extractedImage.extent.size.height))
            self.preview.imageView.image = UIImage(ciImage: scaledImage)
            
            
            let handler = VNImageRequestHandler(ciImage: scaledImage)
            do {
                let request = VNCoreMLRequest(model: self.ocrModel!)
                try handler.perform([request])
                guard let results = request.results as? [VNClassificationObservation] else {
                    return
                }
                
                self.predictionLabel.label.text = "\"\(results[0].identifier)\" \(Int(results[0].confidence * 10.0))"
                
            } catch {
                print(error)
            }
            
        }
    }
    
    func goToNextObservation() {
        if observationsToProcess.count > 0 {
            observationsToProcess.remove(at: 0)
        }
        showCurrentObservation()
    }
    
    func clearObservations() {
        observationsToProcess.removeAll()
        lastImageUsedForObservations = nil
    }

    
    func playCameraImage(_ cameraCaptureHelper: CameraCaptureHelper, image: CIImage, originalImage: CIImage, frameNumber:Int, fps:Int) {
        if frameNumber > 10 && frameNumber % 200 == 0 {
            if overrideImage != nil {
                lastOriginalImage = overrideImage!
            } else {
                lastOriginalImage = originalImage
            }
            
            if observationsToProcess.count == 0 {
                
                DispatchQueue.main.async {
                    self.preview.imageView.image = UIImage(ciImage: self.lastOriginalImage!)
                }

                
                //let convertedImage = image |> adjustColors |> convertToGrayscale
                let handler = VNImageRequestHandler(ciImage: self.lastOriginalImage!)
                let request: VNDetectTextRectanglesRequest =
                    VNDetectTextRectanglesRequest(completionHandler: { [unowned self] (request, error) in
                        if (error != nil) {
                            print("Got Error In Run Text Dectect Request :(")
                        } else {
                            guard let results = request.results as? Array<VNTextObservation> else {
                                fatalError("Unexpected result type from VNDetectTextRectanglesRequest")
                            }
                            if (results.count == 0) {
                                return
                            }
                            
                            self.clearObservations()
                            
                            self.lastImageUsedForObservations = self.lastOriginalImage
                            
                            var avgAspect:CGFloat = 0
                            var numAspect:CGFloat = 0
                            for textObservation in results {
                                for rectangleObservation in textObservation.characterBoxes! {
                                    let charW = rectangleObservation.bottomRight.x - rectangleObservation.bottomLeft.x
                                    let charH = rectangleObservation.topRight.y - rectangleObservation.bottomRight.y
                                    let charAspect = charW/charH
                                    
                                    // sanity assumption for single character aspect
                                    if charAspect < 1.4 {
                                        avgAspect += charAspect
                                        numAspect += 1.0
                                    }
                                }
                            }
                            avgAspect /= numAspect
                            
                            
                            for textObservation in results {
                                for rectangleObservation in textObservation.characterBoxes! {
                                    
                                    // TODO: try and detect when the vision detected more than one character in
                                    // a box. We can try this simply by checking what the aspect ratio of the ractangle
                                    // is; if its too wide perhaps it is 2 or 3 characters in the box
                                    
                                    let w = self.lastOriginalImage!.extent.width
                                    let h = self.lastOriginalImage!.extent.height
                                    
                                    
                                    let charW = rectangleObservation.bottomRight.x - rectangleObservation.bottomLeft.x
                                    let charH = rectangleObservation.topRight.y - rectangleObservation.bottomRight.y
                                    let charAspect = charW/charH
                                    
                                    let subdivisions = Int(round(charAspect / avgAspect))
                                    
                                    //print("\(subdivisions) \(charAspect)        \(avgAspect)")
                                    
                                    var charX = rectangleObservation.bottomLeft.x
                                    
                                    let deltaW = charW / CGFloat(subdivisions)
                                    
                                    for _ in 0..<subdivisions {
                                        
                                        let perspectiveImagesCoords = [
                                            "inputTopLeft":CIVector(x:(charX + 0) * w, y: rectangleObservation.topLeft.y * h),
                                            "inputTopRight":CIVector(x:(charX + deltaW) * w, y: rectangleObservation.topRight.y * h),
                                            "inputBottomLeft":CIVector(x:(charX + 0) * w, y: rectangleObservation.bottomLeft.y * h),
                                            "inputBottomRight":CIVector(x:(charX + deltaW) * w, y: rectangleObservation.bottomRight.y * h),
                                            ]
                                        
                                        self.observationsToProcess.append(perspectiveImagesCoords)
                                        
                                        charX += deltaW
                                    }
                                }
                                
                                self.observationsToProcess.append(nil)
                            }
                            
                            self.showCurrentObservation()
                        }
                    })
                request.reportCharacterBoxes = true
                do {
                    try handler.perform([request])
                } catch {
                    print(error)
                }
                
            }

            
        }
    }
    
    func loadModels() {
        do {
            
            
            // The OCR model converts image of character into a character
            let modelURL = URL(fileURLWithPath: String(bundlePath:"bundle://Assets/predict/ocr.mlmodel"))
            let compiledUrl = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledUrl)
            self.ocrModel = try? VNCoreMLModel(for: model)
            
            
        } catch {
            print(error)
        }
    }
    
    
    var currentOverrideImageIndex = 0
    
    override func viewDidLoad() {
        title = "Train"
        mainBundlePath = "bundle://Assets/train/train.xml"
        loadView()
        
        overrideImage = CIImage(contentsOf: URL(fileURLWithPath: String(bundlePath: "bundle://Assets/predict/debug/IMG_0001.JPG")))
        
        if(overrideImage != nil) {
            navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Next", style: .plain, target: self, action: #selector(nextOverrideImage))
        }

        super.viewDidLoad()
        
        // this is slightly complex so here it goes:
        // 1. the camera helper should be running barebones by itself
        // 2. we should start another thread which is responsible for processing a single image
        // 3. when that thread is done it waits for the camera helper to supply a new image, the processes that image
        
        captureHelper.delegate = self
        captureHelper.delegateWantsPlayImages = true
        
        UIApplication.shared.isIdleTimerDisabled = true
        
        loadModels()
        
        // debug to save the current image
        let tap = UITapGestureRecognizer(target: self, action: #selector(self.SaveImages(_:)))
        preview.view.addGestureRecognizer(tap)
        preview.view.isUserInteractionEnabled = true
        
        nextObservation.button.add(for: .touchUpInside) {
            self.goToNextObservation()
        }
        
        // Generate all of the class buttons we need and space them nicely in the buttonsContainer
        for i in 0..<ALPHABET.count {
            _ = CreateButton(UInt8(i), ALPHABET[i...i])
        }
        
    }
    
    func CreateButton(_ classValue:UInt8, _ label:String) -> Button {
        
        var leftMargin = 5
        let topMargin = 5
        
        let sizeOfButton = 58
        let buttonPadding = 4
        let buttonsPerRow = (Int(view.frame.width) - leftMargin * 2) / sizeOfButton
        let btnIdx = buttonsContainer.view.subviews.count
        let xIdx = btnIdx % buttonsPerRow
        let yIdx = btnIdx / buttonsPerRow
        
        leftMargin += (Int(view.frame.width) - buttonsPerRow * sizeOfButton) / 2
        
        let btn = Button().new("ButtonStd") { this in
            this.title = label
            
            this.frame = CGRect(x: buttonPadding + leftMargin + xIdx * sizeOfButton, y: buttonPadding + topMargin + yIdx * sizeOfButton, width: sizeOfButton - buttonPadding * 2, height: sizeOfButton - buttonPadding * 2)
        }
        
        btn.button.add(for: .touchUpInside) {
            // submit the image and correct labelling to the server
            var filename = String.init(format: "%@", UUID().uuidString)
            
            for i in 0..<self.ALPHABET.count {
                filename += String.init(format: "_%d", (btnIdx == i ? 1 : 0))
            }
            
            filename += ".png"
            
            let extractedImage = self.lastOriginalImage!.applyingFilter("CIPerspectiveCorrection", parameters: self.observationsToProcess[0] as! [String : Any])
            let scaledImage = extractedImage.transformed(by: CGAffineTransform.init(scaleX: 28.0 / extractedImage.extent.size.width, y: 28.0 / extractedImage.extent.size.height))
            guard let pngData = self.ciContext.pngRepresentation(of:scaledImage, format: kCIFormatARGB8, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]) else {
                return
            }
            
            var dataPacket = Data()
            dataPacket.append(contentsOf: filename.asciiArray8 as [UInt8])
            dataPacket.append(0)
            dataPacket.append(pngData)
            try! self.trainingImagesPublisher?.send(data: dataPacket)
            
            self.goToNextObservation()
        }

        buttonsContainer.view.addSubview(btn.button)
        
        return btn
    }
    
    @objc func nextOverrideImage() {
        
        do {
            let debugUrl = URL(fileURLWithPath: String(bundlePath: "bundle://Assets/predict/debug/"))

            let fileURLs = try FileManager.default.contentsOfDirectory(at: debugUrl, includingPropertiesForKeys: nil)
            
            currentOverrideImageIndex += 1
            if currentOverrideImageIndex > fileURLs.count {
                currentOverrideImageIndex = 0
            }
            
            overrideImage = CIImage(contentsOf: fileURLs[currentOverrideImageIndex])
            
        } catch {
            print("Error going to next debug image")
        }
        
        clearObservations()
    }
    
    @objc func SaveImages(_ sender: UITapGestureRecognizer) {
         let originalImage = ciContext.createCGImage(lastOriginalImage!, from: lastOriginalImage!.extent)
         UIImageWriteToSavedPhotosAlbum(UIImage(cgImage: originalImage!), self, #selector(self.image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
            let ac = UIAlertController(title: "Save error", message: error.localizedDescription, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        } else {
            let ac = UIAlertController(title: "Saved!", message: "Your image has been saved to your photos.", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        UIApplication.shared.isIdleTimerDisabled = false
        captureHelper.stop()
    }
    
    fileprivate var preview: ImageView {
        return mainXmlView!.elementForId("preview")!.asImageView!
    }
    fileprivate var nextObservation: Button {
        return mainXmlView!.elementForId("nextObservation")!.asButton!
    }
    fileprivate var predictionLabel: Label {
        return mainXmlView!.elementForId("predictionLabel")!.asLabel!
    }
    fileprivate var buttonsContainer: View {
        return mainXmlView!.elementForId("buttonsContainer")!.asView!
    }
}

