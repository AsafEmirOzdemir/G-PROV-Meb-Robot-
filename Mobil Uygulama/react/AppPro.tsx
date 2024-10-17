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
const App = (navigator) => {
  const [user,userChange]=useState("Eren Kına");
  return (
    <ImageBackground source={require("./sa.jpg")} resizeMode="cover" style={{width:"100%"}}>
    <SafeAreaView className="flex h-screen bg-asaf" style={{}} >
      
    <View className=" flex items-center justify-center align-middle" style={{backgroundColor:"#06090F", height:60}}>
        <Pressable className="absolute top-2 left-1.5" onPress={()=>{navigator.navigation.navigate("login")}}>
        <Image source={require("./logo.png")} style={{
              margin:4,
              resizeMode: 'cover',
              height: 40,
              width: 40,
            }} className=""></Image></Pressable>
        <View className="flex items-center p-1 m-1 justify-center align-middle">

<Text className="text-white text-xl top-2 pb-2 font-ArchivoBlack_Regular" style={{height:40}}>{user}
</Text>
</View>
      </View>
      <LinearGradient colors={["#06090F","#060910"]} className="h-5"></LinearGradient>
      <View className="flex-grow w-screen flex">
        <View className=" flex-grow  items-center w-screen">
          <Image source={require('./body.png')}></Image>
          <View className="flex-1 w-screen items-center">
          <Text className="text-center font-ArchivoBlack_Regular text-blue-600 pb-4">
                        Egzersiz No: 5
                    </Text>
          
          <View style={{height: '61%', width: '90%'}} className=" border-4 border-blue-600">
          <ScrollViewIndicator scrollViewProps={{overScrollMode:"never",backgroundColor:"#222428",paddingRight:10,paddingLeft:10}}  indStyle={{width:5, backgroundColor:"#bdbdbd", marginRight:2.5}}>
            
                    <View className="">
                    <Text className="text-center font-ArchivoBlack_Regular text-blue-600 pb-2">
                        Üst Vücut Eklem Verileri:
                    </Text>
                    <Text className="font-ArchivoBlack_Regular text-white">
                      Baş: <Text className="text-blue-600">%98</Text> {"\n"}
                      Sağ Omuz:<Text className="text-blue-600"> %97</Text> -- Sol Omuz:<Text className="text-blue-600"> %97</Text>{"\n"}
                      Sağ Dirsek: <Text className="text-blue-600">%88</Text> -- Sol Dirsek <Text className="text-blue-600">%92</Text>{"\n"}
                      Sağ El Bileği: <Text className="text-blue-600">%73</Text> -- Sol El Bileği <Text className="text-blue-600">%97</Text>{"\n"}
                    </Text>
                    <Text className="pb-2 text-center font-ArchivoBlack_Regular text-blue-600">
                        Sağ El Parmak Verileri:
                    </Text>
                    <Text className="font-ArchivoBlack_Regular text-white">
                      Baş Parmak: <Text className="text-blue-600">%95 %97</Text> {"\n"}
                      İşaret Parmağı: <Text className="text-blue-600">%90 %98 %95 </Text>{"\n"}
                      Orta Parmağı: <Text className="text-blue-600">%60 %97 %88</Text>  {"\n"}
                      Yüzük Parmağı: <Text className="text-blue-600">%100 %97 %98</Text> {"\n"}
                      Serçe Parmağı: <Text className="text-blue-600">%97 %98 %97</Text> {"\n"}
                    </Text>
                    <Text className="pb-2 text-center font-ArchivoBlack_Regular text-blue-600">
                        Sol El Parmak Verileri:
                    </Text>
                    <Text className="font-ArchivoBlack_Regular text-white">
                      Baş Parmak: <Text className="text-blue-600">%95 %97</Text> {"\n"}
                      İşaret Parmağı: <Text className="text-blue-600">%90 %98 %95</Text> {"\n"}
                      Orta Parmağı: <Text className="text-blue-600">%60 %97 %88</Text>  {"\n"}
                      Yüzük Parmağı: <Text className="text-blue-600">%100 %97 %98</Text> {"\n"}
                      Serçe Parmağı: <Text className="text-blue-600">%97 %98 %97</Text> {"\n"}
                    </Text>
                    <Text className="pb-2 text-center font-ArchivoBlack_Regular text-blue-600">
                        Alt Vücut Eklem Verileri:
                    </Text>
                    <Text className="font-ArchivoBlack_Regular text-white">
                      Sağ Kalça: <Text className="text-blue-600">%94</Text> -- Sol Kalça: <Text className="text-blue-600">%67</Text> {"\n"}
                      Sağ Diz: <Text className="text-blue-600">%90</Text> -- Sol Diz: <Text className="text-blue-600">%54</Text> {"\n"}
                      Sağ Ayak Bileği: <Text className="text-blue-600">%84</Text> -- Sol Ayak Bileği: <Text className="text-blue-600">%34</Text>
                    </Text>
                    </View>
                </ScrollViewIndicator>
                </View>
                <View className="flex items-center w-screen">
                <Pressable style={{width:"90%"}} className=" justify-center p-2 m-2 bg-blue-700 rounded-sm"  onPress={()=>{navigator.navigation.navigate("protez")}}>
                  <Text className="text-center text-white font-ArchivoBlack_Regular text-lg">
                    Protez Analizlerini Gör
                  </Text>
                </Pressable>
                <Pressable style={{width:"90%"}} className=" justify-center p-2 bg-blue-700 rounded-sm"  onPress={()=>{navigator.navigation.navigate("cam")}}>
                  <Text className="text-center text-white font-ArchivoBlack_Regular text-lg">
                    Egzersiz Seç
                  </Text>
                </Pressable>
                
                </View>
                
                
                
                
        </View>
        </View>
      </View>
      
    </SafeAreaView>
    </ImageBackground>
  );
}
 
export default App;