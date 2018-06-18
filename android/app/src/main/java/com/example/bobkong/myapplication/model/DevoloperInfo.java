package com.example.bobkong.myapplication.model;

import android.graphics.Bitmap;

import java.io.Serializable;

/**
 * Created by Bob on 2018/6/16.
 */

public class DevoloperInfo {
    private String mName;
    private String mJob;
    private int mImage;

    public DevoloperInfo(String name, String job, int image) {
        mName = name;
        mJob = job;
        mImage = image;
    }

    public String getName() {
        return mName;
    }

    public void setName(String name) {
        mName = name;
    }

    public String getJob() {
        return mJob;
    }

    public void setJob(String job) {
        mJob = job;
    }

    public int getImage() {
        return mImage;
    }

    public void setImage(int image) {
        mImage = image;
    }
}
