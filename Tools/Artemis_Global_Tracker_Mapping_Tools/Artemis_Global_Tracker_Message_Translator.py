#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message translator for binary SBD messages

Written by: Andreas Schneider
7th May 2021
20th August 2022

License: MIT

This code translates binary SBD messages by the Artemis Global Tracker.
Binary messages can be read from local files or from email attachments on an IMAP server.
Optionally, the coordinates of all messages can be written into a GPX file.
Alternatively, binary messages can be encoded and written to file or sent to a device.
"""

import numpy as np
from enum import Enum
import datetime
import struct
import gpxpy
import gpxpy.gpx
import configparser
import imaplib
import email
import requests
import os.path
import argparse


class TrackerMessageFields(Enum):
    """
    Define the message field IDs used in the binary format MO and MT messages.
    Taken over from Tracker_Message_Fields.h
    """
    STX       = 0x02
    ETX       = 0x03
    SWVER     = 0x04
    SOURCE    = 0x08
    BATTV     = 0x09
    PRESS     = 0x0a
    TEMP      = 0x0b
    HUMID     = 0x0c
    YEAR      = 0x0d
    MONTH     = 0x0e
    DAY       = 0x0f
    HOUR      = 0x10
    MIN       = 0x11
    SEC       = 0x12
    MILLIS    = 0x13
    DATETIME  = 0x14
    LAT       = 0x15
    LON       = 0x16
    ALT       = 0x17
    SPEED     = 0x18
    HEAD      = 0x19
    SATS      = 0x1a
    PDOP      = 0x1b
    FIX       = 0x1c
    GEOFSTAT  = 0x1d
    USERVAL1  = 0x20
    USERVAL2  = 0x21
    USERVAL3  = 0x22
    USERVAL4  = 0x23
    USERVAL5  = 0x24
    USERVAL6  = 0x25
    USERVAL7  = 0x26
    USERVAL8  = 0x27
    MOFIELDS  = 0x30
    FLAGS1    = 0x31
    FLAGS2    = 0x32
    DEST      = 0x33
    HIPRESS   = 0x34
    LOPRESS   = 0x35
    HITEMP    = 0x36
    LOTEMP    = 0x37
    HIHUMID   = 0x38
    LOHUMID   = 0x39
    GEOFNUM   = 0x3a
    GEOF1LAT  = 0x3b
    GEOF1LON  = 0x3c
    GEOF1RAD  = 0x3d
    GEOF2LAT  = 0x3e
    GEOF2LON  = 0x3f
    GEOF2RAD  = 0x40
    GEOF3LAT  = 0x41
    GEOF3LON  = 0x42
    GEOF3RAD  = 0x43
    GEOF4LAT  = 0x44
    GEOF4LON  = 0x45
    GEOF4RAD  = 0x46
    WAKEINT   = 0x47
    ALARMINT  = 0x48
    TXINT     = 0x49
    LOWBATT   = 0x4a
    DYNMODEL  = 0x4b
    RBHEAD    = 0x52
    USERFUNC1 = 0x58
    USERFUNC2 = 0x59
    USERFUNC3 = 0x5a
    USERFUNC4 = 0x5b
    USERFUNC5 = 0x5c
    USERFUNC6 = 0x5d
    USERFUNC7 = 0x5e
    USERFUNC8 = 0x5f  

"""
Define the type of the binary data fields.
Either a dtype or a length in bytes for more complicated cases.
Entries according to Tracker_Message_Fields.h
"""
FIELD_TYPE = {
        TrackerMessageFields.STX: 0,
        TrackerMessageFields.ETX: 0,
        TrackerMessageFields.SWVER: np.dtype('uint8'),
        TrackerMessageFields.SOURCE: np.dtype('uint32'),
        TrackerMessageFields.BATTV: np.dtype('uint16'),
        TrackerMessageFields.PRESS: np.dtype('uint16'),
        TrackerMessageFields.TEMP: np.dtype('int16'),
        TrackerMessageFields.HUMID: np.dtype('uint16'),
        TrackerMessageFields.YEAR: np.dtype('uint16'),
        TrackerMessageFields.MONTH: np.dtype('uint8'),
        TrackerMessageFields.DAY: np.dtype('uint8'),
        TrackerMessageFields.HOUR: np.dtype('uint8'),
        TrackerMessageFields.MIN: np.dtype('uint8'),
        TrackerMessageFields.SEC: np.dtype('uint8'),
        TrackerMessageFields.MILLIS: np.dtype('uint16'),
        TrackerMessageFields.DATETIME: 7,
        TrackerMessageFields.LAT: np.dtype('int32'),
        TrackerMessageFields.LON: np.dtype('int32'),
        TrackerMessageFields.ALT: np.dtype('int32'),
        TrackerMessageFields.SPEED: np.dtype('int32'),
        TrackerMessageFields.HEAD: np.dtype('int32'),
        TrackerMessageFields.SATS: np.dtype('uint8'),
        TrackerMessageFields.PDOP: np.dtype('uint16'),
        TrackerMessageFields.FIX: np.dtype('uint8'),
        TrackerMessageFields.GEOFSTAT: (np.dtype('uint8'), 3),
        TrackerMessageFields.USERVAL1: np.dtype('uint8'),
        TrackerMessageFields.USERVAL2: np.dtype('uint8'),
        TrackerMessageFields.USERVAL3: np.dtype('uint16'),
        TrackerMessageFields.USERVAL4: np.dtype('uint16'),
        TrackerMessageFields.USERVAL5: np.dtype('uint32'),
        TrackerMessageFields.USERVAL6: np.dtype('uint32'),
        TrackerMessageFields.USERVAL7: np.dtype('float32'),
        TrackerMessageFields.USERVAL8: np.dtype('float32'),
        TrackerMessageFields.MOFIELDS: (np.dtype('uint32'), 3),
        TrackerMessageFields.FLAGS1: np.dtype('uint8'),
        TrackerMessageFields.FLAGS2: np.dtype('uint8'),
        TrackerMessageFields.DEST: np.dtype('uint32'),
        TrackerMessageFields.HIPRESS: np.dtype('uint16'),
        TrackerMessageFields.LOPRESS: np.dtype('uint16'),
        TrackerMessageFields.HITEMP: np.dtype('int16'),
        TrackerMessageFields.LOTEMP: np.dtype('int16'),
        TrackerMessageFields.HIHUMID: np.dtype('uint16'),
        TrackerMessageFields.LOHUMID: np.dtype('uint16'),
        TrackerMessageFields.GEOFNUM: np.dtype('uint8'),
        TrackerMessageFields.GEOF1LAT: np.dtype('int32'),
        TrackerMessageFields.GEOF1LON: np.dtype('int32'),
        TrackerMessageFields.GEOF1RAD: np.dtype('uint32'),
        TrackerMessageFields.GEOF2LAT: np.dtype('int32'),
        TrackerMessageFields.GEOF2LON: np.dtype('int32'),
        TrackerMessageFields.GEOF2RAD: np.dtype('uint32'),
        TrackerMessageFields.GEOF3LAT: np.dtype('int32'),
        TrackerMessageFields.GEOF3LON: np.dtype('int32'),
        TrackerMessageFields.GEOF3RAD: np.dtype('uint32'),
        TrackerMessageFields.GEOF4LAT: np.dtype('int32'),
        TrackerMessageFields.GEOF4LON: np.dtype('int32'),
        TrackerMessageFields.GEOF4RAD: np.dtype('uint32'),
        TrackerMessageFields.WAKEINT: np.dtype('uint32'),
        TrackerMessageFields.ALARMINT: np.dtype('uint16'),
        TrackerMessageFields.TXINT: np.dtype('uint16'),
        TrackerMessageFields.LOWBATT: np.dtype('uint16'),
        TrackerMessageFields.DYNMODEL: np.dtype('uint8'),
        TrackerMessageFields.RBHEAD: 4,
        TrackerMessageFields.USERFUNC1: 0,
        TrackerMessageFields.USERFUNC2: 0,
        TrackerMessageFields.USERFUNC3: 0,
        TrackerMessageFields.USERFUNC4: 0,
        TrackerMessageFields.USERFUNC5: np.dtype('uint16'),
        TrackerMessageFields.USERFUNC6: np.dtype('uint16'),
        TrackerMessageFields.USERFUNC7: np.dtype('uint32'),
        TrackerMessageFields.USERFUNC8: np.dtype('uint32')
}

"""
Conversion factors for data fields according to documentation.
"""
CONVERSION_FACTOR = {
        TrackerMessageFields.BATTV: 1e-2,
        TrackerMessageFields.TEMP: 1e-2,
        TrackerMessageFields.HUMID: 1e-2,
        TrackerMessageFields.LAT: 1e-7,
        TrackerMessageFields.LON: 1e-7,
        TrackerMessageFields.ALT: 1e-3,
        TrackerMessageFields.HEAD: 1e-7,
        TrackerMessageFields.PDOP: 1e-2,
        TrackerMessageFields.HITEMP: 1e-2,
        TrackerMessageFields.LOTEMP: 1e-2,
        TrackerMessageFields.HIHUMID: 1e-2,
        TrackerMessageFields.LOHUMID: 1e-2,
        TrackerMessageFields.GEOF1LAT: 1e-7,
        TrackerMessageFields.GEOF1LON: 1e-7,
        TrackerMessageFields.GEOF1RAD: 1e-2,
        TrackerMessageFields.GEOF2LAT: 1e-7,
        TrackerMessageFields.GEOF2LON: 1e-7,
        TrackerMessageFields.GEOF2RAD: 1e-2,
        TrackerMessageFields.GEOF3LAT: 1e-7,
        TrackerMessageFields.GEOF3LON: 1e-7,
        TrackerMessageFields.GEOF3RAD: 1e-2,
        TrackerMessageFields.GEOF4LAT: 1e-7,
        TrackerMessageFields.GEOF4LON: 1e-7,
        TrackerMessageFields.GEOF4RAD: 1e-2,
        TrackerMessageFields.LOWBATT: 1e-2
}

def checksum(data):
    """
    Compute checksum bytes are as defined in the 8-Bit Fletcher Algorithm,
    used by the TCP standard (RFC 1145).

    Args:
        data, byte array of data

    Return:
        cs_a, checksum a
        cs_b, checksum b
    """
    old_settings = np.seterr(over='ignore') # Do not warn for desired overflow in this computation.
    cs_a = np.uint8(0)
    cs_b = np.uint8(0)
    for ind in range(len(data)):
        cs_a += np.uint8(data[ind])
        cs_b += cs_a
    np.seterr(**old_settings)
    return cs_a, cs_b

def decode_message(message):
    """
    Parse binary SBD message from Sparkfun Artemis Global Tracker.

    Args:
        message, binary message as byte array

    Returns:
        data, translated message as dictionary
    """
    data = {}
    if message[0] == TrackerMessageFields.STX.value:
        ind = 0
    else: # assuming gateway header
        ind = 5
    assert (message[ind] == TrackerMessageFields.STX.value), 'STX marker not found.'
    ind += 1
    while (message[ind] != TrackerMessageFields.ETX.value):
        field = TrackerMessageFields(message[ind])
        ind += 1
        if isinstance(FIELD_TYPE[field], int): # length in bytes
            field_len = FIELD_TYPE[field]
            if field == TrackerMessageFields.DATETIME:
                values = struct.unpack('HBBBBB', message[ind:ind+field_len])
                data[field.name] = datetime.datetime(*values)
            elif field_len > 0:
                data[field.name] = message[ind:ind+field_len]
            else:
                data[field.name] = None
        elif isinstance(FIELD_TYPE[field], np.dtype): # dtype of scalar
            field_len = FIELD_TYPE[field].itemsize
            data[field.name] = np.frombuffer(message[ind:ind+field_len], dtype=FIELD_TYPE[field])[0]
        elif isinstance(FIELD_TYPE[field], tuple): # dtype and length of array
            field_len = FIELD_TYPE[field][0].itemsize * FIELD_TYPE[field][1]
            data[field.name] = np.frombuffer(message[ind:ind+field_len], dtype=FIELD_TYPE[field][0])
        else:
            raise ValueError('Unknown entry in FIELD_TYPE list: {}'.format(FIELD_TYPE[field]))
        if field in CONVERSION_FACTOR:
            data[field.name] = float(data[field.name]) * CONVERSION_FACTOR[field]
        ind += field_len
    ind += 1 # ETX
    cs_a, cs_b = checksum(message[:ind])
    assert (message[ind] == cs_a), 'Checksum mismatch.'
    assert (message[ind+1] == cs_b), 'Checksum mismatch.'
    return data

def encode_message(data):
    """
    Create a binary SBD message in Sparkfun Artemis Global Tracker format.

    Args:
        data, dictionary with data to send, with keys named according to TrackerMessageFields names

    Returns:
        msg, encoded binary SBD message
    """
    msg = b''
    msg += np.uint8(TrackerMessageFields.STX.value)
    for field in TrackerMessageFields:
        if field.name in data:
            msg += np.uint8(field.value)
            if field == TrackerMessageFields.DATETIME:
                msg += struct.pack(
                        'HBBBBB',
                        data[field.name].year, data[field.name].month,
                        data[field.name].day, data[field.name].hour,
                        data[field.name].minute, data[field.name].second)
            elif isinstance(FIELD_TYPE[field], np.dtype): # dtype of scalar
                rawvalue = np.array([ data[field.name] ])
                if field in CONVERSION_FACTOR:
                    rawvalue /= CONVERSION_FACTOR[field]
                msg += rawvalue.astype(FIELD_TYPE[field]).tobytes()
                del rawvalue
            elif isinstance(FIELD_TYPE[field], tuple): # dtype and length of array
                assert(len(data[field.name]) == FIELD_TYPE[field][1])
                rawdata = np.array(data[field.name])
                if field in CONVERSION_FACTOR:
                    rawdata /= CONVERSION_FACTOR[field]
                msg += rawdata.astype(FIELD_TYPE[field]).tobytes()
                del rawdata
            elif isinstance(FIELD_TYPE[field], int): # number of bytes
                msg += np.zeros(FIELD_TYPE[field], dtype=np.uint8).tobytes()
    msg += np.uint8(TrackerMessageFields.ETX.value)
    cs_a, cs_b = checksum(msg)
    msg += cs_a
    msg += cs_b
    return msg

def asc2bin(msg_asc):
    """
    Convert ASCII representation of binary message to real binary message.
    """
    msg = b''
    for ind in np.arange(0,len(msg_asc),2):
        msg += np.uint8(int(msg_asc[ind:ind+2], 16))
    return msg

def bin2asc(msg_bin):
    """
    Encode binary message to ASCII representation.
    """
    msg_asc = ''
    for ind in range(len(msg_bin)):
        msg_asc += '{:02x}'.format(msg_bin[ind])
    return msg_asc

def message2trackpoint(msg):
    """
    Creates a GPX trackpoint from a translated IRIDIUM message.

    Args:
        msg, translated SBD message

    Returns:
        pkt, GPX trackpoint corresponding to message data
    """
    return gpxpy.gpx.GPXTrackPoint(
                msg['LAT'], msg['LON'], elevation=msg['ALT'], time=msg['DATETIME'],
                comment='{} hPa'.format(msg['PRESS']) if 'PRESS' in msg else None)

def query_mail(imap, from_address='@rockblock.rock7.com', unseen_only=True):
    """
    Query IMAP server for new mails from IRIDIUM gateway and extract new messages.

    Args:
        imap, imaplib object with open connection
        from_address, sender address to filter for
        unseen_only, whether to only retrieve unseen messages (default: True)

    Returns:
        sbd_list, list of sbd attachments
    """
    sbd_list = []
    imap.select('Inbox')
    criteria = ['FROM', from_address]
    if unseen_only:
        criteria.append('(UNSEEN)')
    retcode, messages = imap.search(None, *criteria)
    for num in messages[0].split():
        typ, data = imap.fetch(num, '(RFC822)')
        raw_message = data[0][1]
        message = email.message_from_bytes(raw_message)
        # Download attachments
        for part in message.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            filename = part.get_filename()
            if bool(filename):
                _, fileext = os.path.splitext(filename)
                if fileext in ['.sbd', '.bin']:
                    sbd_list.append(part.get_payload(decode=True))
                else:
                    print('query_mail: unrecognized file extension {} of attachment.'.format(fileext))
    return sbd_list

def get_messages(imap, from_address='@rockblock.rock7.com', all_messges=False):
    """
    Get IRIDIUM SBD messages from IMAP.

    Args:
        imap, imaplib object with open connection
        from_address, sender address to filter for
        unseen_only, whether to only retrieve unseen messages (default: True)

    Returns:
        messages, translated messages as a list of dictionaries
    """
    messages = []
    sbd_list = query_mail(imap, from_address=from_address, unseen_only=not all_messges)
    for sbd in sbd_list:
        try:
            messages.append(decode_message(sbd))
        except (ValueError, AssertionError) as err:
            print('Error translating message: ',err)
            pass
    return messages

def send_message(imei, data, user, password):
    """
    Send a mobile terminated (MT) message to a RockBLOCK device.

    Args:
        imei, the destination IMEI number
        data, the data to send
        user, the RockBLOCK username
        password, the RockBLOCK password

    Returns:
        success, True on success, else False
        message, an error or success message
    """
    resp = requests.post(
            'https://core.rock7.com/rockblock/MT',
            data={'imei': imei, 'data': bin2asc(data),
                  'username': user, 'password': password})
    if not resp.ok:
        print('Error sending message: POST command failed: {}'.format(resp.text))
    parts = resp.text.split(',')
    if parts[0] == 'OK':
        try:
            print('Message {} sent.'.format(parts[1]))
        except IndexError:
            print('Unexpected server response format.')
        return True, 'OK'
    elif parts[0] == 'FAILED':
        error_message = ''
        try:
            print('Sending message failed with error code {}: {}'.format(parts[1], parts[2]))
            error_message = parts[2]
        except IndexError:
            print('Unexpected server response format.')
        return False, error_message
    else:
        error_message = 'Unexpected server response: {}'.format(resp.text)
        print(error_message)
        return False, error_message

def write_gpx(gpx_track, output_file):
    """
    Write a track to a GPX file.

    Args:
        gpx_track, the track to be written
        output_file, the file name to be written
    """
    gpx = gpxpy.gpx.GPX()
    gpx.creator = 'AGT Message Translator'
    gpx.name = 'Artemis Global Tracker'
    gpx.tracks.append(gpx_track)
    with open(output_file, 'w') as fd:
        fd.write(gpx.to_xml())
    return

def main_decode(filelist, use_imap=False, all_messages=False, output_file=None):
    """
    Decode binary SBD messages and write out data.
    """
    if output_file:
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
    if use_imap:
        config = configparser.ConfigParser()
        config.read(filelist[0])
        hostname = config['email']['host']
        print('Connecting to {} ...'.format(hostname))
        imap = imaplib.IMAP4_SSL(hostname) # connect to host using SSL
        imap.login(config['email']['user'], config['email']['password']) # login to server
        messages = get_messages(imap, from_address=config['email'].get('from', fallback='@rockblock.rock7.com'), all_messges=all_messages)
        imap.close()
        for msg in messages:
            print(msg)
            if output_file:
                gpx_segment.points.append(message2trackpoint(msg))
    else:
        if len(filelist) == 1 and not os.path.isfile(filelist[0]):
            # Argument is supposed to be an ASCII representation of a binary message.
            msg_bin = asc2bin(filelist[0])
            msg_trans = decode_message(msg_bin)
            print(msg_trans)
        else:
            for filename in filelist:
                with open(filename,'rb') as fd:
                    msg_bin = fd.read()
                    try:
                        msg_trans = decode_message(msg_bin)
                    except (ValueError, AssertionError, IndexError) as err:
                        print('Error translating message {}: {}'.format(filename, err))
                        continue
                    print(filename, msg_trans)
                    if output_file:
                        gpx_segment.points.append(message2trackpoint(msg_trans))
    if output_file:
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_track.segments.append(gpx_segment)
        print('Writing {}'.format(output_file))
        write_gpx(gpx_track, output_file)
    return

def main_encode(position=None, time=None, userfunc=None, output_file=None, send=None):
    """
    Encode binary SBD message corresponding to given data
    and write it to file or send it to mobile IRIDIUM device.
    """
    data = {}
    if position is not None:
        data.update({
            'LON': position[0],
            'LAT': position[1],
            'ALT': position[2]})
    if time:
        data.update({'DATETIME': time})
    if userfunc:
        data.update(userfunc)
    message = encode_message(data)
    if output_file:
        with open(output_file, 'wb') as fd:
            fd.write(message)
    else:
        print(bin2asc(message))
    if send:
        config = configparser.ConfigParser()
        config.read(send)
        status, error_message = send_message(
                config['device']['imei'], message,
                config['rockblock']['user'], config['rockblock']['password'])
        if status:
            print('Message sent.')
        else:
            print('Error sending message: {}'.format(error_message))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--decode', nargs='+', help='Decode messages from files or argument')
    parser.add_argument('-i', '--imap', required=False, action='store_true', default=False, help='Query imap server instead of reading local files. filenames argument will be interpreted as ini file.')
    parser.add_argument('-a', '--all', required=False, action='store_true', default=False, help='Retrieve all messages, not only unread ones. Only relevant in combination with -i.')
    parser.add_argument('-o', '--output', required=False, default=None, help='Optional output GPX file')
    parser.add_argument('-e', '--encode', required=False, action='store_true', default=False, help='Encode binary message')
    parser.add_argument('-p', '--position', required=False, default=None, help='Position lon,lat,alt')
    parser.add_argument('-t', '--time', required=False, nargs='?', default=None, const=datetime.datetime.utcnow(), help='UTC time in ISO format YYYY-mm-dd HH:MM:SS, or now if no argument given')
    parser.add_argument('-u', '--userfunc', required=False, default=None, help='User function (comma-separated list of functions to trigger)')
    parser.add_argument('-s', '--send', required=False, default=None, help='Send message to device as specified in configuration file')
    args = parser.parse_args()
    if args.encode:
        if args.position is not None:
            position = np.array(args.position.split(',')).astype(float)
        else:
            position = None
        if args.time is not None:
            if isinstance(args.time, datetime.datetime):
                time = args.time
            else:
                time = datetime.datetime.fromisoformat(args.time)
        else:
            time = None
        if args.userfunc is not None:
            userfunc = {}
            funcs = args.userfunc.split(',')
            for func in funcs:
                if ':' in func:
                    vals = func.split(':')
                    userfunc.update({'USERFUNC'+vals[0]: vals[1]})
                else:
                    userfunc.update({'USERFUNC'+func: True})
        else:
            userfunc = None
        main_encode(position=position, time=time, userfunc=userfunc, output_file=args.output, send=args.send)
    elif args.decode:
        if args.imap:
            assert (len(args.decode) == 1), 'In combination with the -i option, exactly one file name must be given, namely the ini file.'
        main_decode(args.decode, use_imap=args.imap, all_messages=args.all, output_file=args.output)
    else:
        print('Decode or encode needs to be specified.')
