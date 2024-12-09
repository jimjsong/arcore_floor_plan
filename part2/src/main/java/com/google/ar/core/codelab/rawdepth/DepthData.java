/*
 * Copyright 2021 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.codelab.rawdepth;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.Image;
import android.opengl.Matrix;
import android.util.Log;

import com.google.ar.core.Anchor;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.exceptions.NotYetAvailableException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.HashMap;

/**
 * Converts depth data from ARCore depth images to 3D pointclouds. Points are added by calling the
 * Raw Depth API, and reprojected into 3D space.
 */
public class DepthData {
    public static final int FLOATS_PER_POINT = 4; // X,Y,Z,confidence.

    public static float f_min_x = Float.MAX_VALUE, f_min_y = Float.MAX_VALUE, f_min_z = Float.MAX_VALUE,
            f_max_x = Float.MIN_VALUE, f_max_y = Float.MIN_VALUE, f_max_z = Float.MIN_VALUE;

    private static HashMap<String, Float> heightMap = new HashMap<>();

    public static HashMap<String, Float> getHeightMap() {
        return heightMap;
    }
    public static FloatBuffer create(Frame frame, Anchor cameraPoseAnchor, float maxFloorHeight) {
        try {
            Image depthImage = frame.acquireRawDepthImage16Bits();
            Image confidenceImage = frame.acquireRawDepthConfidenceImage();

            // To transform 2D depth pixels into 3D points we retrieve the intrinsic camera parameters
            // corresponding to the depth image. See more information about the depth values at
            // https://developers.google.com/ar/develop/java/depth/overview#understand-depth-values.
            final CameraIntrinsics intrinsics = frame.getCamera().getTextureIntrinsics();
            float[] modelMatrix = new float[16];
            cameraPoseAnchor.getPose().toMatrix(modelMatrix, 0);
            final FloatBuffer points = convertRawDepthImagesTo3dPointBuffer(
                    depthImage, confidenceImage, intrinsics, modelMatrix, maxFloorHeight);

            depthImage.close();
            confidenceImage.close();

            return points;
        } catch (NotYetAvailableException e) {
            // This normally means that depth data is not available yet. This is normal so we will not
            // spam the logcat with this.
        }
        return null;
    }

    /** Applies camera intrinsics to convert depth image into a 3D pointcloud. */
    private static FloatBuffer convertRawDepthImagesTo3dPointBuffer(
            Image depth, Image confidence, CameraIntrinsics cameraTextureIntrinsics, float[] modelMatrix, float maxFloorHeight) {
        // Java uses big endian so we have to change the endianess to ensure we extract
        // depth data in the correct byte order.
        final Image.Plane depthImagePlane = depth.getPlanes()[0];
        ByteBuffer depthByteBufferOriginal = depthImagePlane.getBuffer();
        ByteBuffer depthByteBuffer = ByteBuffer.allocate(depthByteBufferOriginal.capacity());
        depthByteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        while (depthByteBufferOriginal.hasRemaining()) {
            depthByteBuffer.put(depthByteBufferOriginal.get());
        }
        depthByteBuffer.rewind();
        ShortBuffer depthBuffer = depthByteBuffer.asShortBuffer();

        final Image.Plane confidenceImagePlane = confidence.getPlanes()[0];
        ByteBuffer confidenceBufferOriginal = confidenceImagePlane.getBuffer();
        ByteBuffer confidenceBuffer = ByteBuffer.allocate(confidenceBufferOriginal.capacity());
        confidenceBuffer.order(ByteOrder.LITTLE_ENDIAN);
        while (confidenceBufferOriginal.hasRemaining()) {
            confidenceBuffer.put(confidenceBufferOriginal.get());
        }
        confidenceBuffer.rewind();

        // To transform 2D depth pixels into 3D points we retrieve the intrinsic camera parameters
        // corresponding to the depth image. See more information about the depth values at
        // https://developers.google.com/ar/develop/java/depth/overview#understand-depth-values.
        final int[] intrinsicsDimensions = cameraTextureIntrinsics.getImageDimensions();
        final int depthWidth = depth.getWidth();
        final int depthHeight = depth.getHeight();
        final float fx =
                cameraTextureIntrinsics.getFocalLength()[0] * depthWidth / intrinsicsDimensions[0];
        final float fy =
                cameraTextureIntrinsics.getFocalLength()[1] * depthHeight / intrinsicsDimensions[1];
        final float cx =
                cameraTextureIntrinsics.getPrincipalPoint()[0] * depthWidth / intrinsicsDimensions[0];
        final float cy =
                cameraTextureIntrinsics.getPrincipalPoint()[1] * depthHeight / intrinsicsDimensions[1];

        // Allocate the destination point buffer. If the number of depth pixels is larger than
        // `maxNumberOfPointsToRender` we uniformly subsample. The raw depth image may have
        // different resolutions on different devices.
        final float maxNumberOfPointsToRender = 20000;
        int step = (int) Math.ceil(Math.sqrt(depthWidth * depthHeight / maxNumberOfPointsToRender));

        FloatBuffer points = FloatBuffer.allocate(depthWidth / step * depthHeight / step * FLOATS_PER_POINT);
        float[] pointCamera = new float[4];
        float[] pointWorld = new float[4];

        for (int y = 0; y < depthHeight; y += step) {
            for (int x = 0; x < depthWidth; x += step) {
                // Depth images are tightly packed, so it's OK to not use row and pixel strides.
                int depthMillimeters = depthBuffer.get(y * depthWidth + x); // Depth image pixels are in mm.
                if (depthMillimeters == 0) {
                    // Pixels with value zero are invalid, meaning depth estimates are missing from
                    // this location.
                    continue;
                }
                final float depthMeters = depthMillimeters / 1000.0f; // Depth image pixels are in mm.

                // Retrieves the confidence value for this pixel.
                final byte confidencePixelValue =
                        confidenceBuffer.get(
                                y * confidenceImagePlane.getRowStride()
                                        + x * confidenceImagePlane.getPixelStride());
                final float confidenceNormalized = ((float) (confidencePixelValue & 0xff)) / 255.0f;

                // Unprojects the depth into a 3D point in camera coordinates.
                pointCamera[0] = depthMeters * (x - cx) / fx;
                pointCamera[1] = depthMeters * (cy - y) / fy;
                pointCamera[2] = -depthMeters;
                pointCamera[3] = 1;

                // Applies model matrix to transform point into world coordinates.
                Matrix.multiplyMV(pointWorld, 0, modelMatrix, 0, pointCamera, 0);

                if (pointWorld[1] < maxFloorHeight * .95  && confidenceNormalized > .3) {
                    float f_x = pointWorld[0];
                    float f_y = pointWorld[1];
                    float f_z = pointWorld[2];

                    if (f_x < f_min_x) {
                        f_min_x = f_x;
                    }
                    if (f_x > f_max_x) {
                        f_max_x = f_x;
                    }
                    if (f_y < f_min_y) {
                        f_min_y = f_y;
                    }
                    if (f_y > f_max_y) {
                        f_max_y = f_y;
                    }
                    if (f_z < f_min_z) {
                        f_min_z = f_z;
                    }
                    if (f_z > f_max_z) {
                        f_max_z = f_z;
                    }

                    int i_x = (int) (f_x * 100 / 40);
                    int i_z = (int) (f_z * 100 / 40);

                    String key = i_x + "," + i_z;
                    heightMap.put(key, f_y);

                    points.put(f_x); // X.
                    points.put(f_y); // Y.
                    points.put(f_z); // Z.
                    points.put(confidenceNormalized);
                }

            }
            Log.d("DepthData",
                    "f_min_x: " + f_min_x + ", f_min_y: " + f_min_y + ", f_min_z: " + f_min_z +
                            ", f_max_x: " + f_max_x + ", f_max_y: " + f_max_y + ", f_max_z: " + f_max_z);
        }

        Log.d("DepthData", "Number of points: " + points.limit() / FLOATS_PER_POINT);
        points.rewind();
        return points;
    }

    public static Bitmap createHeightMapBitmap() {
        Bitmap bitmap = Bitmap.createBitmap(400, 400, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        canvas.drawColor(Color.TRANSPARENT); // Clear background

        // Calculate scale factors
        float scaleX = 400f / (f_max_x - f_min_x);
        float scaleZ = 400f / (f_max_z - f_min_z);

        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.FILL);
        paint.setStrokeWidth(5f); // Make points larger

        for (HashMap.Entry<String, Float> entry : heightMap.entrySet()) {
            String[] coords = entry.getKey().split(",");
            int i_x = Integer.parseInt(coords[0]);
            int i_z = Integer.parseInt(coords[1]);
            float f_y = entry.getValue();

            // Convert world coordinates to bitmap coordinates
            float x = (i_x * 40f / 100f - f_min_x) * scaleX;
            float z = (i_z * 40f / 100f - f_min_z) * scaleZ;

            // Ensure points are within bounds
            x = Math.min(Math.max(x, 0), 399);
            z = Math.min(Math.max(z, 0), 399);

            // Draw a circle instead of a point for better visibility
            canvas.drawCircle(x, 399 - z, 3f, paint);
        }

        return bitmap;
    }
}
