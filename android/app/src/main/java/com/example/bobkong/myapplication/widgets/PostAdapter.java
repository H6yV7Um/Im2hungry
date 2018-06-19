package com.example.bobkong.myapplication.widgets;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;

import com.example.bobkong.myapplication.R;
import com.example.bobkong.myapplication.model.PostDataManager;
import com.example.bobkong.myapplication.model.PostInfo;

import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Created by bobkong on 2018/6/9.
 */

public class PostAdapter extends BaseAdapter {

    List<PostInfo> mDatas = new ArrayList<>();
    LayoutInflater mInflater;
    Context mContext;

    public PostAdapter(Context context) {
        if (context == null) {
            return;
        }
        mContext = context;
        mInflater = LayoutInflater.from(context);
    }

    public List<PostInfo> getDatas() {
        return mDatas;
    }

    public void setDatas(List<PostInfo> datas) {
        mDatas = datas;
        notifyDataSetChanged();
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        if (position < 0 || position >= mDatas.size()) {
            return null;
        }
        PostViewHolder postViewHolder;
        if (convertView != null) {
            postViewHolder = (PostViewHolder) convertView.getTag();
        } else {
            convertView = mInflater.inflate(R.layout.post_item, null);
            postViewHolder = new PostViewHolder(convertView, mContext);
            convertView.setTag(postViewHolder);
        }

        try {
            postViewHolder.PaintView(mDatas.get(position));
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return convertView;
    }

    @Override
    public int getCount() {
        return mDatas.size();
    }

    @Override
    public Object getItem(int position) {
        return null;
    }

    @Override
    public long getItemId(int position) {
        // 获取当前数据的hashCode
        int hashCode = mDatas.get(position).hashCode();
        return hashCode;
    }

}
