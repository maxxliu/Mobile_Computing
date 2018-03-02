/* LCM type definition class file
 * This file was automatically generated by lcm-gen
 * DO NOT MODIFY BY HAND!!!!
 */

package mcomp_msgs;
 
import java.io.*;
import java.util.*;
import lcm.lcm.*;
 
public final class event_msg implements lcm.lcm.LCMEncodable
{
    public long timestamp;
    public double position[];
    public String description;
 
    public event_msg()
    {
        position = new double[3];
    }
 
    public static final long LCM_FINGERPRINT;
    public static final long LCM_FINGERPRINT_BASE = 0xc7e9bfc12f308bfeL;
 
    static {
        LCM_FINGERPRINT = _hashRecursive(new ArrayList<Class<?>>());
    }
 
    public static long _hashRecursive(ArrayList<Class<?>> classes)
    {
        if (classes.contains(mcomp_msgs.event_msg.class))
            return 0L;
 
        classes.add(mcomp_msgs.event_msg.class);
        long hash = LCM_FINGERPRINT_BASE
            ;
        classes.remove(classes.size() - 1);
        return (hash<<1) + ((hash>>63)&1);
    }
 
    public void encode(DataOutput outs) throws IOException
    {
        outs.writeLong(LCM_FINGERPRINT);
        _encodeRecursive(outs);
    }
 
    public void _encodeRecursive(DataOutput outs) throws IOException
    {
        char[] __strbuf = null;
        outs.writeLong(this.timestamp); 
 
        for (int a = 0; a < 3; a++) {
            outs.writeDouble(this.position[a]); 
        }
 
        __strbuf = new char[this.description.length()]; this.description.getChars(0, this.description.length(), __strbuf, 0); outs.writeInt(__strbuf.length+1); for (int _i = 0; _i < __strbuf.length; _i++) outs.write(__strbuf[_i]); outs.writeByte(0); 
 
    }
 
    public event_msg(byte[] data) throws IOException
    {
        this(new LCMDataInputStream(data));
    }
 
    public event_msg(DataInput ins) throws IOException
    {
        if (ins.readLong() != LCM_FINGERPRINT)
            throw new IOException("LCM Decode error: bad fingerprint");
 
        _decodeRecursive(ins);
    }
 
    public static mcomp_msgs.event_msg _decodeRecursiveFactory(DataInput ins) throws IOException
    {
        mcomp_msgs.event_msg o = new mcomp_msgs.event_msg();
        o._decodeRecursive(ins);
        return o;
    }
 
    public void _decodeRecursive(DataInput ins) throws IOException
    {
        char[] __strbuf = null;
        this.timestamp = ins.readLong();
 
        this.position = new double[(int) 3];
        for (int a = 0; a < 3; a++) {
            this.position[a] = ins.readDouble();
        }
 
        __strbuf = new char[ins.readInt()-1]; for (int _i = 0; _i < __strbuf.length; _i++) __strbuf[_i] = (char) (ins.readByte()&0xff); ins.readByte(); this.description = new String(__strbuf);
 
    }
 
    public mcomp_msgs.event_msg copy()
    {
        mcomp_msgs.event_msg outobj = new mcomp_msgs.event_msg();
        outobj.timestamp = this.timestamp;
 
        outobj.position = new double[(int) 3];
        System.arraycopy(this.position, 0, outobj.position, 0, 3); 
        outobj.description = this.description;
 
        return outobj;
    }
 
}
