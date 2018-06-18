package com.example.bobkong.myapplication.model;

import android.graphics.Bitmap;

import com.example.bobkong.myapplication.net.model.User;
import com.google.gson.annotations.SerializedName;

import java.io.Serializable;

/**
 * Created by bobkong on 2018/6/7.
 */
public class PostInfo implements Serializable{
    @SerializedName("post_id")
    private String postId;
    @SerializedName("user")
    private User mUser;

    @SerializedName("food_cal")
    private int mCal;
    @SerializedName("food_name")
    private String mFoodName;
    @SerializedName("loc_lat")
    private double mLocLat;
    @SerializedName("loc_lng")
    private double mLocLng;
    @SerializedName("loc_name")
    private String mLocName;
    @SerializedName("description")
    private String mDescription;
    @SerializedName("post_time")
    private String mPostTime;
    @SerializedName("pic_url")
    private String mPostImageUrl;

    @SerializedName("favored")
    private boolean mFavored;
    @SerializedName("favoriteCount")
    private int mFavorNum;

    public boolean isFavored() {
        return mFavored;
    }

    public void setFavored(boolean favored) {
        mFavored = favored;
    }


    public PostInfo(User user, int mCal, String mFoodName, double mLocLat, double mLocLng, String mLocName, String mDescription, String mPostTime, String mPostImageUrl, int mFavorNum) {
        this.mCal = mCal;
        mUser = user;
        this.mFoodName = mFoodName;
        this.mLocLat = mLocLat;
        this.mLocLng = mLocLng;
        this.mLocName = mLocName;
        this.mDescription = mDescription;
        this.mPostTime = mPostTime;
        this.mPostImageUrl = mPostImageUrl;
        this.mFavorNum = mFavorNum;
    }

    public PostInfo() {
    }

    public String getPostId() {
        return postId;
    }

    public void setPostId(String postId) {
        this.postId = postId;
    }

    public User getUser() {
        return mUser;
    }

    public void setUser(User user) {
        mUser = user;
    }

    public void setCal(int cal) {
        mCal = cal;
    }

    public void setFoodName(String foodName) {
        mFoodName = foodName;
    }

    public void setLocLat(double locLat) {
        mLocLat = locLat;
    }

    public void setLocLng(double locLng) {
        mLocLng = locLng;
    }

    public void setLocName(String locName) {
        mLocName = locName;
    }

    public void setDescription(String description) {
        mDescription = description;
    }

    public void setPostTime(String postTime) {
        mPostTime = postTime;
    }

    public void setFavorNum(int favorNum) {
        mFavorNum = favorNum;
    }

    public String getUserName() {
        return mUser.getUserName();
    }

    public String getUserImage() {
        return mUser.getUserImg();
    }

    public int getCal() {
        return mCal;
    }

    public String getFoodName() {
        return mFoodName;
    }

    public double getLocLat() {
        return mLocLat;
    }

    public double getLocLng() {
        return mLocLng;
    }

    public String getLocName() {
        return mLocName;
    }

    public String getDescription() {
        return mDescription;
    }

    public String getPostTime() {
        return mPostTime;
    }

    public String getPostImageUrl() {
        return mPostImageUrl;
    }

    public int getFavorNum() {
        return mFavorNum;
    }

    public void setPostImageUrl(String picture){
        this.mPostImageUrl = picture;
    }
}
