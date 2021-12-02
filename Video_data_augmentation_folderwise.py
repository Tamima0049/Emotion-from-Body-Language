#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from moviepy.editor import *


def process_video(input, output):
    """Parameter input should be a string with the full path for a video"""

    clip = VideoFileClip(input)
    clip1 = clip.fx(vfx.colorx, 0.6) #1.1 means 10% bright, 0.1 means 10% dark

    #clip1 = clip.rotate(270)

    #clip1 = clip.fx(vfx.mirror_x)

    #clip3 = clip2.resize(width=1920)
    #clip1 = clip.fx(vfx.crop, y2 = 325)

    clip1.write_videofile(output, codec='rawvideo')


def get_video_paths(folder_path):
    """ 
    Parameter folder_path should look like "Users/documents/folder1/"
    Returns a list of complete paths
    """
    file_name_list = os.listdir(folder_path)

    path_name_list = []
    final_name_list = []
    for name in file_name_list:
        # Put any sanity checks here, e.g.:
        if name == ".DS_Store":
            pass
        else:
            path_name_list.append(folder_path + name)
            final_name_list.append(folder_path + "40%dark" + name )
    return path_name_list, final_name_list


# In[2]:


if __name__ == "__main__":
    video_folder = "/Users/tamima_rashid/Desktop/Data/IRB/IRB_New/p19_cropped/"
    path_list, final_name_list = get_video_paths(video_folder)
    for path, name in zip(path_list, final_name_list):
        process_video(path, name)
    print("Finished")


# In[ ]:




