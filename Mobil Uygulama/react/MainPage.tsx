import { View , Text} from "react-native";
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import Login from "./LoginPage";
import App from "./AppPro";
import Protez from "./protez";
import Cam from "./camera";

const Stack = createNativeStackNavigator();
const Main = () => {
    return (
        <NavigationContainer>
            <Stack.Navigator initialRouteName="login" screenOptions={{animation:"slide_from_left",animationDuration:100}}>
                <Stack.Screen
                name="login"
                component={Login}
                options={{headerShown:false}}>
                </Stack.Screen>
                <Stack.Screen
                name="userpage"
                component={App}
                options={{headerShown:false}}
                >
                </Stack.Screen>
                <Stack.Screen
                name="protez"
                component={Protez}
                options={{headerShown:false}}
                >
                </Stack.Screen>
                <Stack.Screen
                name="cam"
                component={Cam}
                options={{headerShown:false}}
                >
                </Stack.Screen>
            </Stack.Navigator>
        </NavigationContainer>
    );
}
 
export default Main;