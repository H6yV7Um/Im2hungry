package com.example.bobkong.myapplication.widgets;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.amap.api.location.AMapLocation;
import com.amap.api.location.AMapLocationClient;
import com.amap.api.location.AMapLocationClientOption;
import com.amap.api.location.AMapLocationListener;
import com.bumptech.glide.Glide;
import com.example.bobkong.myapplication.R;
import com.example.bobkong.myapplication.model.PostDataManager;
import com.example.bobkong.myapplication.model.PostInfo;
import com.example.bobkong.myapplication.net.FavoriteService;
import com.example.bobkong.myapplication.net.LocationService;
import com.example.bobkong.myapplication.router.RouterHelper;
import com.example.bobkong.myapplication.tools.CalculateDistance;
import com.example.bobkong.myapplication.tools.FormatCurrentData;
import com.example.bobkong.myapplication.ui.MyPostActivity;
import com.jaren.lib.view.LikeView;


import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

import rx.android.schedulers.AndroidSchedulers;
import rx.schedulers.Schedulers;

/**
 * Created by bobkong on 2018/6/8.
 */

public class PostViewHolder {
    private static final String LOG_TAG = "POST_VIEWHOLDER";
    private ImageView userImage;
    private TextView userName;
    private TextView description;
    private ImageView postImage;
    private TextView locationName;
    private TextView distance;
    private TextView foodName;
    private TextView cal;
    private TextView favorNum;
    private TextView postTime;
    private Context mContext;
    private PostInfo mPostInfo;
    private LikeView favor_button;
    private boolean mChecked;

    public PostViewHolder(View view, Context context) {
        mContext = context;
        userImage = view.findViewById(R.id.user_image);
        userName = view.findViewById(R.id.user_name);
        description = view.findViewById(R.id.description);
        foodName = view.findViewById(R.id.food_name);
        cal = view.findViewById(R.id.cal);
        postImage = view.findViewById(R.id.post_image);
        locationName = view.findViewById(R.id.loc_name);
        distance = view.findViewById(R.id.distance);
        favorNum = view.findViewById(R.id.favor_num);
        postTime = view.findViewById(R.id.post_time);


        favor_button = (LikeView) view.findViewById(R.id.favor);


        locationName.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                RouterHelper.IntentToLocationSeeActivity(mContext, mPostInfo.getLocLat(), mPostInfo.getLocLng());
            }
        });

    }

    public void PaintView(PostInfo postInfo) throws ParseException {
        if (postInfo == null) {
            return;
        }
        mPostInfo = postInfo;
        Glide.with(mContext).load(Uri.parse(postInfo.getUserImage())).into(userImage);
        userName.setText(postInfo.getUserName());
        description.setText(postInfo.getDescription());
        Glide.with(mContext).load(Uri.parse(postInfo.getPostImageUrl())).placeholder(R.mipmap.post_image_placeholder).into(postImage);
        locationName.setText(postInfo.getLocName());
        distance.setText(getDistance());
        foodName.setText(postInfo.getFoodName());
        cal.setText(postInfo.getCal() + "cal/kg");
        favorNum.setText(postInfo.getFavorNum() + "");
        postTime.setText(FormatCurrentData.getTimeRange(postInfo.getPostTime()));

        if (postInfo.isFavored()) {
            favor_button.setChecked(true);
            mChecked = true;
        } else {
            favor_button.setChecked(false);
            mChecked = false;
        }

        favor_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mChecked) {
                    FavoriteService.getFavoriteService().favor(mPostInfo.getPostId(), false)
                            .observeOn(AndroidSchedulers.mainThread())
                            .subscribe(response -> {
                                Log.d(LOG_TAG, "unfavor: " + response.toString());
                                if (response.isSuccess()) {
                                    mPostInfo.setFavorNum(mPostInfo.getFavorNum() - 1);
                                    favorNum.setText(String.valueOf(mPostInfo.getFavorNum()));
                                    Log.d(LOG_TAG, "unfavor favor num = " + mPostInfo.getFavorNum());
                                }
                            }, Throwable::printStackTrace);
                } else {
                    FavoriteService.getFavoriteService().favor(mPostInfo.getPostId(), true)
                            .observeOn(AndroidSchedulers.mainThread())
                            .subscribe(response -> {
                                Log.d(LOG_TAG, "favor: " + response.toString());
                                if (response.isSuccess()) {
                                    mPostInfo.setFavorNum(mPostInfo.getFavorNum() + 1);
                                    favorNum.setText(String.valueOf(mPostInfo.getFavorNum()));
                                    Log.d(LOG_TAG, "favor num = " + mPostInfo.getFavorNum());
                                }
                            }, Throwable::printStackTrace);
                }
                mChecked = !mChecked;
                favor_button.setChecked(mChecked);
            }
        });
        userImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mContext == null) {
                    return;
                }
                RouterHelper.IntentToMyPostActivity(mContext, MyPostActivity.MY_POST, postInfo.getUser());
            }
        });

    }


    private String getDistance() {

        double distance = CalculateDistance.algorithm(mPostInfo.getLocLng(), mPostInfo.getLocLat(), LocationService.getInstance().getLng(), LocationService.getInstance().getLat());
        if (distance >= 1000) {
            return (String.valueOf(distance / 1000) + "km");
        }
        return String.valueOf(distance) + "m";
    }

}