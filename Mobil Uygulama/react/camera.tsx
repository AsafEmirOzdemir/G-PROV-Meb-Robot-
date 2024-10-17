import { useState } from "react";
import LinearGradient from "react-native-linear-gradient";
import { ScrollView, StyleSheet } from "react-native";
import {ScrollViewIndicator} from '@fanchenbao/react-native-scroll-indicator';
const style=StyleSheet.create({
  text:{
    fontFamily:"ArchivoBlack-Regular"
  }
})
import{
  View,
  Text,
  Image,
  SafeAreaView,
  Alert,
  Touchable,
  ImageBackground,
  Pressable,
} from "react-native"
const Cam = (navigator) => {
  
  const alert=()=>{
    Alert.alert("Egzersiz Seçildi !", "GİPROV GÖRÜNTÜ İŞLEME uygulamasını açınız.")
  }
  return (

    <ImageBackground className="flex w-screen h-screen"  source={require("./sa.jpg")} resizeMode="cover">
      <View className=" flex items-center justify-center align-middle" style={{backgroundColor:"#06090F", height:60}}>
        <Pressable className="absolute top-2 left-1.5" onPress={()=>{navigator.navigation.navigate("userpage")}}>
        <Image source={require("./logo.png")} style={{
              margin:4,
              resizeMode: 'cover',
              height: 40,
              width: 40,
            }} className=""></Image></Pressable>
        <View className="flex items-center p-1 m-1 justify-center align-middle">

<Text className="text-white text-xl top-2 pb-2 font-ArchivoBlack_Regular" style={{height:40}}>Eren Kına İçin Egzersizler
</Text>
</View>
</View>
<View style={{height:"90%"}}>
      <ScrollViewIndicator className="h-screen flex-grow flex" style={{flex:1}} scrollViewProps={{overScrollMode:"never"}}  indStyle={{width:3, backgroundColor:"#bdbdbd", marginRight:3}}>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className=" text-xl text-white text-center font-ArchivoBlack_Regular">Egzersiz 1</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl text-white text-center font-ArchivoBlack_Regular">Egzersiz 2</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 3</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="font-ArchivoBlack_Regular text-xl text-white text-center">Egzersiz 4</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="font-ArchivoBlack_Regular text-xl text-white text-center">Egzersiz 5</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="font-ArchivoBlack_Regular text-xl text-white text-center">Egzersiz 6</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 7</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="font-ArchivoBlack_Regular text-xl text-white text-center">Egzersiz 8</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 9</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 10</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 11</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="font-ArchivoBlack_Regular text-xl text-white text-center">Egzersiz 12</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-white text-center text-xl font-ArchivoBlack_Regular">Egzersiz 13</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="font-ArchivoBlack_Regular text-xl text-white text-center">Egzersiz 14</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 15</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 16</Text>
      </Pressable>
      <Pressable className="bg-blue-600 p-2 m-2" onPress={alert}>
        <Text className="text-xl font-ArchivoBlack_Regular text-white text-center">Egzersiz 17</Text>
      </Pressable>
      
    </ScrollViewIndicator>
    </View>
    </ImageBackground>
  );
};

export default Cam;
