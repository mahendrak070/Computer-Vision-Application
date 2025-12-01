// Tracker implementations for different modes

class ObjectTracker {
    constructor() {
        this.trackingMode = 'marker';
        this.isTracking = false;
        this.template = null;
        this.templateRect = null;
        this.sam2Data = null;
        this.arucoDetector = null;
        this.qrDetector = null;
        this.tracker = null;
        
        this.initDetectors();
    }
    
    initDetectors() {
        // Initialize ArUco detector - check if module is available
        try {
            if (typeof cv.aruco !== 'undefined' && typeof cv.aruco.getPredefinedDictionary === 'function') {
                const dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250);
                const parameters = new cv.aruco.DetectorParameters();
                this.arucoDetector = new cv.aruco.ArucoDetector(dictionary, parameters);
                console.log('ArUco detector initialized');
            } else if (typeof cv.getPredefinedDictionary === 'function') {
                // Try alternative API
                const dictionary = cv.getPredefinedDictionary(cv.DICT_6X6_250);
                const parameters = new cv.aruco_DetectorParameters();
                this.arucoDetector = new cv.aruco_ArucoDetector(dictionary, parameters);
                console.log('ArUco detector initialized (alternative API)');
            } else {
                // ArUco not available - will use contour-based fallback
                console.log('ArUco module not available - using contour-based marker detection');
            }
        } catch (e) {
            console.warn('ArUco detector initialization failed:', e);
            this.arucoDetector = null;
        }
        
        // Initialize QR Code detector - check if module is available
        try {
            if (typeof cv.QRCodeDetector !== 'undefined') {
                this.qrDetector = new cv.QRCodeDetector();
                console.log('QR Code detector initialized');
            } else if (typeof cv.wechat_qrcode !== 'undefined') {
                // Try wechat QR code detector as alternative
                this.qrDetector = new cv.wechat_qrcode.WeChatQRCode();
                console.log('WeChat QR Code detector initialized');
            } else {
                // QR Code not available - will use contour-based fallback
                console.log('QR Code detector not available - using contour-based marker detection');
            }
        } catch (e) {
            console.warn('QR Code detector initialization failed:', e);
            this.qrDetector = null;
        }
        
        // Initialize feature detector as fallback for marker-less detection
        // Note: Not critical since marker-less tracking uses template matching
        try {
            if (typeof cv.ORB !== 'undefined') {
                // Try different ORB API variations
                if (typeof cv.ORB === 'function') {
                    this.orbDetector = cv.ORB();
                } else if (typeof cv.ORB.create === 'function') {
                    this.orbDetector = cv.ORB.create();
                } else {
                    this.orbDetector = null;
                }
                
                if (this.orbDetector && typeof cv.BFMatcher !== 'undefined') {
                    this.bfMatcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
                    console.log('Feature detector initialized for fallback');
                } else {
                    this.orbDetector = null;
                }
            } else {
                this.orbDetector = null;
            }
        } catch (e) {
            // Not critical - marker-less tracking uses template matching anyway
            this.orbDetector = null;
        }
    }
    
    setMode(mode) {
        this.trackingMode = mode;
        this.reset();
    }
    
    reset() {
        this.template = null;
        this.templateRect = null;
        this.sam2Data = null;
        this.tracker = null;
    }
    
    // Marker-based tracking (ArUco/QR with fallback)
    trackMarker(src, dst) {
        let found = false;
        let count = 0;
        
        // Try ArUco marker detection first
        if (this.arucoDetector) {
            try {
                const cornersVec = new cv.MatVector();
                const idsVec = new cv.MatVector();
                const rejectedVec = new cv.MatVector();
                
                this.arucoDetector.detectMarkers(src, cornersVec, idsVec, rejectedVec);
                
                if (idsVec.size() > 0) {
                    found = true;
                    count = idsVec.size();
                    for (let i = 0; i < cornersVec.size(); i++) {
                        const corners = cornersVec.get(i);
                        this.drawMarker(dst, corners);
                    }
                }
                
                cornersVec.delete();
                idsVec.delete();
                rejectedVec.delete();
            } catch (e) {
                console.warn('ArUco detection error:', e);
            }
        }
        
        // If ArUco not found, try QR code
        if (!found && this.qrDetector) {
            try {
                const points = new cv.Mat();
                const straightQrcode = new cv.Mat();
                
                const detected = this.qrDetector.detectAndDecode(src, points, straightQrcode);
                
                if (detected && points.rows > 0) {
                    found = true;
                    count = 1;
                    this.drawQRCode(dst, points);
                }
                
                points.delete();
                straightQrcode.delete();
            } catch (e) {
                console.warn('QR detection error:', e);
            }
        }
        
        // Fallback: Use contour-based marker detection
        if (!found) {
            const result = this.trackMarkerContours(src, dst);
            found = result.found || false;
            count = result.count || 0;
        }
        
        return { found, count };
    }
    
    // Fallback marker detection using contours and shape recognition
    trackMarkerContours(src, dst) {
        try {
            // Convert to grayscale
            const gray = new cv.Mat();
            if (src.channels() === 3 || src.channels() === 4) {
                cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            } else {
                gray = src.clone();
            }
            
            // Apply threshold to find dark markers
            const thresh = new cv.Mat();
            cv.threshold(gray, thresh, 127, 255, cv.THRESH_BINARY_INV);
            
            // Find contours
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            let found = false;
            let count = 0;
            const borderColor = new cv.Scalar(0, 255, 0); // Bright green border
            const centerColor = new cv.Scalar(255, 0, 255); // Magenta center point
            const cornerColor = new cv.Scalar(0, 255, 255); // Yellow corner points
            
            // Create a mask for overlay
            const mask = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC4);
            const maskColor = new cv.Scalar(0, 255, 100, 100); // Light green with transparency
            
            // Look for square/rectangular markers
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const area = cv.contourArea(contour);
                
                // Filter by area (markers should be reasonably sized)
                if (area > 500 && area < 50000) {
                    // Approximate contour to polygon
                    const epsilon = 0.02 * cv.arcLength(contour, true);
                    const approx = new cv.Mat();
                    cv.approxPolyDP(contour, approx, epsilon, true);
                    
                    // Check if it's roughly square (4 corners)
                    if (approx.rows >= 4 && approx.rows <= 6) {
                        // Get bounding rect
                        const rect = cv.boundingRect(approx);
                        
                        // Check aspect ratio (should be roughly square)
                        const aspectRatio = rect.width / rect.height;
                        if (aspectRatio > 0.7 && aspectRatio < 1.3) {
                            // Create mask overlay for this marker
                            const maskRect = new cv.Rect(rect.x, rect.y, rect.width, rect.height);
                            const roi = mask.roi(maskRect);
                            roi.setTo(maskColor);
                            roi.delete();
                            
                            // Draw filled rectangle for better visibility
                            const pt1 = new cv.Point(rect.x, rect.y);
                            const pt2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
                            
                            // Draw thick border
                            cv.rectangle(dst, pt1, pt2, borderColor, 3);
                            
                            // Draw corner points
                            const corners = [
                                new cv.Point(rect.x, rect.y),
                                new cv.Point(rect.x + rect.width, rect.y),
                                new cv.Point(rect.x + rect.width, rect.y + rect.height),
                                new cv.Point(rect.x, rect.y + rect.height)
                            ];
                            
                            corners.forEach(corner => {
                                cv.circle(dst, corner, 6, cornerColor, -1);
                                cv.circle(dst, corner, 8, cornerColor, 2);
                            });
                            
                            // Draw center point with larger, more visible marker
                            const center = new cv.Point(
                                rect.x + rect.width / 2,
                                rect.y + rect.height / 2
                            );
                            cv.circle(dst, center, 8, centerColor, -1);
                            cv.circle(dst, center, 12, centerColor, 2);
                            
                            // Draw crosshair at center
                            const crossSize = 15;
                            cv.line(dst, 
                                new cv.Point(center.x - crossSize, center.y),
                                new cv.Point(center.x + crossSize, center.y),
                                centerColor, 2);
                            cv.line(dst,
                                new cv.Point(center.x, center.y - crossSize),
                                new cv.Point(center.x, center.y + crossSize),
                                centerColor, 2);
                            
                            found = true;
                            count++;
                        }
                    }
                    approx.delete();
                }
                contour.delete();
            }
            
            // Apply mask overlay if markers found
            if (found) {
                cv.addWeighted(dst, 1.0, mask, 0.3, 0, dst);
            }
            
            // Clean up
            gray.delete();
            thresh.delete();
            contours.delete();
            hierarchy.delete();
            mask.delete();
            
            return { found, count };
        } catch (e) {
            console.error('Contour-based marker detection error:', e);
            return { found: false, count: 0 };
        }
    }
    
    drawMarker(dst, corners) {
        if (!corners || corners.rows === 0) return;
        
        const color = new cv.Scalar(0, 255, 0);
        const thickness = 3;
        
        // Draw marker outline
        for (let i = 0; i < corners.rows; i++) {
            const pt1 = new cv.Point(corners.data32F[i * 8], corners.data32F[i * 8 + 1]);
            const pt2 = new cv.Point(corners.data32F[i * 8 + 2], corners.data32F[i * 8 + 3]);
            const pt3 = new cv.Point(corners.data32F[i * 8 + 4], corners.data32F[i * 8 + 5]);
            const pt4 = new cv.Point(corners.data32F[i * 8 + 6], corners.data32F[i * 8 + 7]);
            
            cv.line(dst, pt1, pt2, color, thickness);
            cv.line(dst, pt2, pt3, color, thickness);
            cv.line(dst, pt3, pt4, color, thickness);
            cv.line(dst, pt4, pt1, color, thickness);
        }
        
        // Draw center
        const centerX = (corners.data32F[0] + corners.data32F[2] + corners.data32F[4] + corners.data32F[6]) / 4;
        const centerY = (corners.data32F[1] + corners.data32F[3] + corners.data32F[5] + corners.data32F[7]) / 4;
        cv.circle(dst, new cv.Point(centerX, centerY), 5, color, -1);
    }
    
    drawQRCode(dst, points) {
        if (!points || points.rows === 0) return;
        
        const color = new cv.Scalar(255, 0, 0);
        const thickness = 3;
        
        // Draw QR code outline
        const pt1 = new cv.Point(points.data32F[0], points.data32F[1]);
        const pt2 = new cv.Point(points.data32F[2], points.data32F[3]);
        const pt3 = new cv.Point(points.data32F[4], points.data32F[5]);
        const pt4 = new cv.Point(points.data32F[6], points.data32F[7]);
        
        cv.line(dst, pt1, pt2, color, thickness);
        cv.line(dst, pt2, pt3, color, thickness);
        cv.line(dst, pt3, pt4, color, thickness);
        cv.line(dst, pt4, pt1, color, thickness);
        
        // Draw center
        const centerX = (points.data32F[0] + points.data32F[2] + points.data32F[4] + points.data32F[6]) / 4;
        const centerY = (points.data32F[1] + points.data32F[3] + points.data32F[5] + points.data32F[7]) / 4;
        cv.circle(dst, new cv.Point(centerX, centerY), 5, color, -1);
    }
    
    // Marker-less tracking using template matching
    setTemplate(rect, frame) {
        this.templateRect = rect;
        const x = Math.max(0, rect.x);
        const y = Math.max(0, rect.y);
        const w = Math.min(rect.width, frame.cols - x);
        const h = Math.min(rect.height, frame.rows - y);
        
        this.template = frame.roi(new cv.Rect(x, y, w, h));
    }
    
    trackMarkerless(src, dst) {
        if (!this.template || this.template.empty()) {
            return false;
        }
        
        try {
            // Convert to grayscale if needed
            let srcGray = new cv.Mat();
            let templateGray = new cv.Mat();
            
            if (src.channels() === 3 || src.channels() === 4) {
                cv.cvtColor(src, srcGray, cv.COLOR_RGBA2GRAY);
            } else {
                srcGray = src.clone();
            }
            
            if (this.template.channels() === 3 || this.template.channels() === 4) {
                cv.cvtColor(this.template, templateGray, cv.COLOR_RGBA2GRAY);
            } else {
                templateGray = this.template.clone();
            }
            
            // Template matching
            const result = new cv.Mat();
            cv.matchTemplate(srcGray, templateGray, result, cv.TM_CCOEFF_NORMED);
            
            const minMax = cv.minMaxLoc(result);
            const maxPoint = minMax.maxLoc;
            const maxVal = minMax.maxVal;
            
            // Threshold for detection
            if (maxVal > 0.6) {
                // Create green mask overlay
                const mask = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC4);
                const maskRect = new cv.Rect(
                    maxPoint.x,
                    maxPoint.y,
                    this.template.cols,
                    this.template.rows
                );
                
                // Fill mask region with light green
                const roi = mask.roi(maskRect);
                const maskColor = new cv.Scalar(0, 255, 100, 100); // Light green with transparency
                roi.setTo(maskColor);
                roi.delete();
                
                // Apply mask overlay
                cv.addWeighted(dst, 1.0, mask, 0.3, 0, dst);
                
                // Draw bounding box using two points (OpenCV.js API)
                const pt1 = new cv.Point(maxPoint.x, maxPoint.y);
                const pt2 = new cv.Point(
                    maxPoint.x + this.template.cols,
                    maxPoint.y + this.template.rows
                );
                
                // Draw bounding box with green styling
                const borderColor = new cv.Scalar(0, 255, 0); // Bright green border
                const centerColor = new cv.Scalar(0, 255, 100); // Light green center
                const cornerColor = new cv.Scalar(100, 255, 0); // Lime green corners
                
                // Draw thick green border
                cv.rectangle(dst, pt1, pt2, borderColor, 3);
                
                // Draw corner points
                const corners = [
                    new cv.Point(maxPoint.x, maxPoint.y),
                    new cv.Point(maxPoint.x + this.template.cols, maxPoint.y),
                    new cv.Point(maxPoint.x + this.template.cols, maxPoint.y + this.template.rows),
                    new cv.Point(maxPoint.x, maxPoint.y + this.template.rows)
                ];
                
                corners.forEach(corner => {
                    cv.circle(dst, corner, 6, cornerColor, -1);
                    cv.circle(dst, corner, 8, cornerColor, 2);
                });
                
                // Draw center point with crosshair
                const center = new cv.Point(
                    maxPoint.x + this.template.cols / 2,
                    maxPoint.y + this.template.rows / 2
                );
                cv.circle(dst, center, 8, centerColor, -1);
                cv.circle(dst, center, 12, centerColor, 2);
                
                // Draw crosshair in bright green
                const crossSize = 15;
                cv.line(dst,
                    new cv.Point(center.x - crossSize, center.y),
                    new cv.Point(center.x + crossSize, center.y),
                    borderColor, 2);
                cv.line(dst,
                    new cv.Point(center.x, center.y - crossSize),
                    new cv.Point(center.x, center.y + crossSize),
                    borderColor, 2);
                
                // Clean up
                mask.delete();
                result.delete();
                srcGray.delete();
                templateGray.delete();
                return true;
            }
            
            result.delete();
            srcGray.delete();
            templateGray.delete();
            return false;
        } catch (e) {
            console.error('Template matching error:', e);
            return false;
        }
    }
    
    // SAM2 segmentation-based tracking
    async loadSAM2Data(npzData) {
        // Store the raw NPZ data
        this.sam2Data = npzData;
        this.sam2Masks = null;
        this.sam2Centroids = [];
        
        // Parse NPZ file
        await this.parseSAM2Data(npzData);
        console.log('SAM2 data loaded');
    }
    
    async parseSAM2Data(arrayBuffer) {
        try {
            // Use NPZ parser to extract data
            const parser = new NPZParser();
            const npz = await parser.parseNPZ(arrayBuffer);
            
            console.log('NPZ file parsed. Keys:', Object.keys(npz));
            
            // Extract masks
            if (npz.masks) {
                const maskArray = npz.masks;
                console.log('Masks shape:', maskArray.shape);
                
                // Convert to OpenCV Mat format
                if (maskArray.shape.length === 3) {
                    // Multiple masks: (N, H, W)
                    this.sam2Masks = parser.numpyToMat(maskArray);
                    console.log(`Loaded ${this.sam2Masks.length} masks`);
                } else if (maskArray.shape.length === 2) {
                    // Single mask: (H, W)
                    this.sam2Masks = [parser.numpyToMat(maskArray)];
                    console.log('Loaded 1 mask');
                } else {
                    console.warn('Unexpected mask shape:', maskArray.shape);
                    this.sam2Masks = [];
                }
            } else {
                console.warn('No masks found in NPZ file');
                this.sam2Masks = [];
            }
            
            // Extract centroids
            if (npz.centroids) {
                const centroidArray = npz.centroids;
                console.log('Centroids shape:', centroidArray.shape);
                
                // Convert to array of {x, y} objects
                if (centroidArray.shape.length === 2 && centroidArray.shape[1] === 2) {
                    this.sam2Centroids = [];
                    const data = centroidArray.data;
                    for (let i = 0; i < centroidArray.shape[0]; i++) {
                        this.sam2Centroids.push({
                            x: data[i * 2],
                            y: data[i * 2 + 1]
                        });
                    }
                    console.log(`Loaded ${this.sam2Centroids.length} centroids`);
                } else {
                    console.warn('Unexpected centroid shape:', centroidArray.shape);
                    this.sam2Centroids = [];
                }
            } else {
                // Compute centroids from masks if not provided
                if (this.sam2Masks && this.sam2Masks.length > 0) {
                    this.sam2Centroids = [];
                    for (let i = 0; i < this.sam2Masks.length; i++) {
                        const centroid = this.computeCentroid(this.sam2Masks[i]);
                        if (centroid) {
                            this.sam2Centroids.push(centroid);
                        }
                    }
                    console.log(`Computed ${this.sam2Centroids.length} centroids from masks`);
                } else {
                    this.sam2Centroids = [];
                }
            }
            
            console.log('SAM2 data structure initialized successfully');
        } catch (e) {
            console.error('SAM2 parsing error:', e);
            this.sam2Masks = [];
            this.sam2Centroids = [];
        }
    }
    
    computeCentroid(mask) {
        try {
            const moments = cv.moments(mask, false);
            
            if (moments.m00 === 0) {
                return null;
            }
            
            const cx = moments.m10 / moments.m00;
            const cy = moments.m01 / moments.m00;
            
            return { x: cx, y: cy };
        } catch (e) {
            console.error('Error computing centroid:', e);
            return null;
        }
    }
    
    trackSAM2(src, dst) {
        if (!this.sam2Data) {
            return { found: false, count: 0 };
        }
        
        try {
            // If we have parsed masks, use them for tracking
            if (this.sam2Masks && this.sam2Masks.length > 0) {
                return this.trackWithMasks(src, dst);
            }
            
            // Otherwise, use a simplified tracking approach
            // This demonstrates the framework - replace with actual mask-based tracking
            return this.trackWithPlaceholder(src, dst);
        } catch (e) {
            console.error('SAM2 tracking error:', e);
            return { found: false, count: 0 };
        }
    }
    
    trackWithMasks(src, dst) {
        // Track objects using actual segmentation masks
        if (!this.sam2Masks || this.sam2Masks.length === 0) {
            return { found: false, count: 0 };
        }
        
        const borderColor = new cv.Scalar(255, 0, 255); // Magenta border
        const centerColor = new cv.Scalar(255, 0, 255); // Magenta center
        const maskColor = new cv.Scalar(255, 0, 255, 100); // Magenta mask with transparency
        let found = false;
        let count = 0;
        
        // Create combined mask overlay
        const combinedMask = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC4);
        
        // For each mask in the SAM2 data
        for (let i = 0; i < this.sam2Masks.length; i++) {
            const mask = this.sam2Masks[i];
            const centroid = this.sam2Centroids[i];
            
            if (!mask || mask.empty() || !centroid) {
                continue;
            }
            
            // Scale mask to match current frame size if needed
            let scaledMask = mask;
            if (mask.rows !== src.rows || mask.cols !== src.cols) {
                scaledMask = new cv.Mat();
                cv.resize(mask, scaledMask, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_LINEAR);
            }
            
            // Convert binary mask to RGBA for overlay
            const maskRGBA = new cv.Mat();
            cv.cvtColor(scaledMask, maskRGBA, cv.COLOR_GRAY2RGBA);
            
            // Apply mask color
            const colorMat = new cv.Mat(maskRGBA.rows, maskRGBA.cols, cv.CV_8UC4, maskColor);
            cv.bitwise_and(maskRGBA, colorMat, maskRGBA);
            
            // Add to combined mask
            cv.add(combinedMask, maskRGBA, combinedMask);
            
            // Compute bounding box from mask
            const rect = cv.boundingRect(scaledMask);
            
            // Draw bounding box using two points
            const pt1 = new cv.Point(rect.x, rect.y);
            const pt2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
            cv.rectangle(dst, pt1, pt2, borderColor, 3);
            
            // Draw corner points
            const corners = [
                new cv.Point(rect.x, rect.y),
                new cv.Point(rect.x + rect.width, rect.y),
                new cv.Point(rect.x + rect.width, rect.y + rect.height),
                new cv.Point(rect.x, rect.y + rect.height)
            ];
            
            corners.forEach(corner => {
                cv.circle(dst, corner, 6, borderColor, -1);
                cv.circle(dst, corner, 8, borderColor, 2);
            });
            
            // Draw centroid with crosshair
            const centerPt = new cv.Point(Math.round(centroid.x), Math.round(centroid.y));
            cv.circle(dst, centerPt, 8, centerColor, -1);
            cv.circle(dst, centerPt, 12, centerColor, 2);
            
            // Draw crosshair
            const crossSize = 15;
            cv.line(dst,
                new cv.Point(centerPt.x - crossSize, centerPt.y),
                new cv.Point(centerPt.x + crossSize, centerPt.y),
                borderColor, 2);
            cv.line(dst,
                new cv.Point(centerPt.x, centerPt.y - crossSize),
                new cv.Point(centerPt.x, centerPt.y + crossSize),
                borderColor, 2);
            
            // Clean up scaled mask if we created it
            if (scaledMask !== mask) {
                scaledMask.delete();
            }
            maskRGBA.delete();
            colorMat.delete();
            
            found = true;
            count++;
        }
        
        // Apply combined mask overlay
        if (found) {
            cv.addWeighted(dst, 1.0, combinedMask, 0.3, 0, dst);
        }
        
        // Clean up
        combinedMask.delete();
        
        return { found, count };
    }
    
    trackWithPlaceholder(src, dst) {
        // Placeholder implementation for demonstration
        // Shows where segmentation-based tracking would work
        const color = new cv.Scalar(255, 0, 255);
        
        // Draw a placeholder region
        // In real implementation, this would be replaced with actual mask rendering
        const x = Math.floor(src.cols * 0.3);
        const y = Math.floor(src.rows * 0.3);
        const w = Math.floor(src.cols * 0.4);
        const h = Math.floor(src.rows * 0.4);
        
        // Create mask overlay
        const mask = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC4);
        const maskRect = new cv.Rect(x, y, w, h);
        const roi = mask.roi(maskRect);
        const maskColor = new cv.Scalar(255, 0, 255, 100); // Magenta mask with transparency
        roi.setTo(maskColor);
        roi.delete();
        
        // Apply mask overlay
        cv.addWeighted(dst, 1.0, mask, 0.3, 0, dst);
        
        // Draw bounding box using two points (OpenCV.js API)
        const pt1 = new cv.Point(x, y);
        const pt2 = new cv.Point(x + w, y + h);
        cv.rectangle(dst, pt1, pt2, color, 3);
        
        // Clean up mask
        mask.delete();
        
        // Draw corner points
        const corners = [
            new cv.Point(x, y),
            new cv.Point(x + w, y),
            new cv.Point(x + w, y + h),
            new cv.Point(x, y + h)
        ];
        
        corners.forEach(corner => {
            cv.circle(dst, corner, 6, color, -1);
            cv.circle(dst, corner, 8, color, 2);
        });
        
        // Draw center point with crosshair
        const center = new cv.Point(
            x + w / 2,
            y + h / 2
        );
        cv.circle(dst, center, 8, color, -1);
        cv.circle(dst, center, 12, color, 2);
        
        // Draw crosshair
        const crossSize = 15;
        cv.line(dst,
            new cv.Point(center.x - crossSize, center.y),
            new cv.Point(center.x + crossSize, center.y),
            color, 2);
        cv.line(dst,
            new cv.Point(center.x, center.y - crossSize),
            new cv.Point(center.x, center.y + crossSize),
            color, 2);
        
        return { found: true, count: 1 };
    }
    
    processFrame(src, dst) {
        let tracked = false;
        let objectCount = 0;
        
        switch (this.trackingMode) {
            case 'marker':
                const markerResult = this.trackMarker(src, dst);
                tracked = markerResult.found || false;
                objectCount = markerResult.count || 0;
                break;
            case 'markerless':
                tracked = this.trackMarkerless(src, dst);
                objectCount = tracked ? 1 : 0;
                break;
            case 'sam2':
                const sam2Result = this.trackSAM2(src, dst);
                tracked = sam2Result.found || false;
                objectCount = sam2Result.count || 0;
                break;
            default:
                tracked = false;
                objectCount = 0;
        }
        
        return { tracked, objectCount };
    }
}

