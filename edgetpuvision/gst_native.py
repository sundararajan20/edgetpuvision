# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cairo
import contextlib
import ctypes
import threading

import gi
gi.require_version('Gdk', '3.0')
gi.require_version('GObject', '2.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gdk, GObject, Gst

Gdk.init([])

# ctypes imports for missing or broken introspection APIs.
libgst = ctypes.CDLL('libgstreamer-1.0.so.0')
libgst.gst_context_writable_structure.restype = ctypes.c_void_p
libgst.gst_context_writable_structure.argtypes = [ctypes.c_void_p]
libgst.gst_structure_set.restype = ctypes.c_void_p
libgst.gst_structure_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

libgdk = ctypes.CDLL('libgdk-3.so.0')
libgdk.gdk_wayland_window_get_wl_surface.restype = ctypes.c_void_p
libgdk.gdk_wayland_window_get_wl_surface.argtypes = [ctypes.c_void_p]
libgdk.gdk_wayland_display_get_wl_display.restype = ctypes.c_void_p
libgdk.gdk_wayland_display_get_wl_display.argtypes = [ctypes.c_void_p]

def set_display_contexts(sink, widget):
    handle = libgdk.gdk_wayland_window_get_wl_surface(hash(widget.get_window()))
    sink.set_window_handle(handle)

    wl_display = libgdk.gdk_wayland_display_get_wl_display(hash(Gdk.Display.get_default()))
    context = Gst.Context.new('GstWaylandDisplayHandleContextType', True)
    structure = libgst.gst_context_writable_structure(hash(context))
    libgst.gst_structure_set(structure, ctypes.c_char_p('display'.encode()),
            hash(GObject.TYPE_POINTER), wl_display, 0)
    sink.set_context(context)

