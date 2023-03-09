package com.android.example.fypapp;


//import com.google.ar.core.Point;

import org.opencv.core.Point;

import org.opencv.core.Rect;

public class Face {
    public Rect faceRect;
    public int faceConfidence;
    public Point[] faceLandmarks;
}
